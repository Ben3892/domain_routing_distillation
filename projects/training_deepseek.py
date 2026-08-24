from typing import List, Optional, Tuple, Union

import torch
import torch._dynamo

from megatron.core import parallel_state
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.enums import ModelType
from megatron.core.transformer import MLATransformerConfig
from megatron.core.transformer.spec_utils import import_module
from megatron.training import get_args, print_rank_0    # 不使用这里的 pretrain
from megatron.training import inprocess_restart
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
)
from megatron_patch.arguments import get_patch_args, validate_args_patched
from megatron_patch.tokenizer import build_tokenizer, get_tokenizer
from megatron_patch.model.lightning_model import LightningModel
from megatron_patch.helper import forward_step
from megatron_patch.training import pretrain
from megatron_patch.data import build_weighted_datasets, build_teacher_datasets

torch._dynamo.config.suppress_errors = True

def _get_transformer_layer_spec(use_te, config):
    """Get transformer layer specification based on configuration.
    
    Args:
        use_te (bool): Whether to use Transformer Engine
        args: Training arguments
        config: Model configuration
        
    Returns:
        transformer_layer_spec: The transformer layer specification
    """
    args = get_args()
    if use_te:
        return get_gpt_layer_with_transformer_engine_spec(
            args.num_experts,
            args.moe_grouped_gemm,
            args.qk_layernorm,
            args.multi_latent_attention,
            args.moe_use_legacy_grouped_gemm,
            qk_l2_norm=args.qk_l2_norm,
            use_kitchen=config.use_kitchen,
        )
    else:
        return get_gpt_layer_local_spec(
            args.num_experts,
            args.moe_grouped_gemm,
            args.qk_layernorm,
            args.multi_latent_attention,
            args.moe_use_legacy_grouped_gemm,
            normalization=args.normalization,
            use_kitchen=config.use_kitchen,
        )


from dataclasses import dataclass
@dataclass
class DeepSeekV2TransformerConfig(MLATransformerConfig):

    moe_ffn_hidden_size: int = None

    moe_layer_freq: int = None

    original_max_position_embeddings: int = 4096
    """Maximum position embeddings for the original model, used by yarn."""

def model_provider(
    pre_process=True, post_process=True, vp_stage: Optional[int] = None
) -> Union[GPTModel]:

    args = get_args()
    build_tokenizer(args)
    config = core_transformer_config_from_args(args, DeepSeekV2TransformerConfig)
    use_te = args.transformer_impl == "transformer_engine"

    if args.spec is not None:
        transformer_layer_spec = import_module(args.spec)
    else:
        if args.num_experts:
            transformer_layer_spec = get_gpt_decoder_block_spec(
                config, use_transformer_engine=use_te, 
                normalization=args.normalization, qk_l2_norm=args.qk_l2_norm, 
                vp_stage=vp_stage
            )
        else:
            if use_te:
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                    args.num_experts, args.moe_grouped_gemm,
                    args.qk_layernorm, args.multi_latent_attention, args.moe_use_legacy_grouped_gemm)
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(
                    args.num_experts, args.moe_grouped_gemm,
                    args.qk_layernorm, args.multi_latent_attention, args.moe_use_legacy_grouped_gemm,
                    normalization=args.normalization)

    mtp_block_spec = None
    if args.mtp_num_layers is not None:
        if hasattr(transformer_layer_spec, 'layer_specs') and len(transformer_layer_spec.layer_specs) == 0:
            transformer_layer_spec_for_mtp = _get_transformer_layer_spec(use_te, config)
        else:
            transformer_layer_spec_for_mtp = transformer_layer_spec
        mtp_block_spec = get_gpt_mtp_block_spec(
            config, transformer_layer_spec_for_mtp, use_transformer_engine=use_te, vp_stage=vp_stage
        )
    
    model = LightningModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True, # Wrapper handles output, but base model should keep it parallel
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor,
        rope_scaling=args.use_rope_scaling,
        mtp_block_spec=mtp_block_spec,
        vp_stage=vp_stage,
    )
    
    return model


def is_dataset_built_on_rank():
    return (
        parallel_state.is_pipeline_first_stage(ignore_virtual=True)
        or parallel_state.is_pipeline_last_stage(ignore_virtual=True)
    ) and parallel_state.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    # Sometimes --data-path is too long, instead we parse it from a file.
    blend: Optional[Tuple[List[str], Optional[List[float]]]]
    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=blend,
        blend_per_split=blend_per_split,
        split=args.split,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
        object_storage_cache_path=args.object_storage_cache_path,
        mid_level_dataset_surplus=args.mid_level_dataset_surplus,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()
    config = core_gpt_dataset_config_from_args(args)
    
    if args.dataset == "PRETRAIN-WITH-WEIGHT":
        print_rank_0("> Building weighted BlendedDataset...")
        train_ds, valid_ds, test_ds = build_weighted_datasets(args)
        print_rank_0("> finished creating weighted BlendedDataset datasets ...")
        return train_ds, valid_ds, test_ds
    elif args.dataset == "PT-TEACHER":
        print_rank_0("> Building weighted BlendedPTTeacherDataset...")
        train_ds, valid_ds, test_ds = build_teacher_datasets(args)
        print_rank_0("> finished creating weighted BlendedPTTeacherDataset datasets ...")
        return train_ds, valid_ds, test_ds
    elif args.sft:
        dataset_type = SFTDataset
    else:
        if args.mock_data:
            dataset_type = MockGPTDataset
        else:
            dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type, train_val_test_num_samples, is_dataset_built_on_rank, config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds

if __name__ == "__main__":
    # 在 Megatron 初始化之前，替换掉原生的参数验证函数
    from megatron.training import initialize
    initialize.validate_args = validate_args_patched

    train_valid_test_datasets_provider.is_distributed = True

    # Optionally enable inprocess restart on pretrain
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=get_patch_args,
    )