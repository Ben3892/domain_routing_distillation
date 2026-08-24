"""
额外的megatron参数添加
"""

import torch
from typing import Union, List
import argparse
import dataclasses
import torch.nn.functional as F
from megatron.core.transformer import TransformerConfig
import json

from megatron.training.arguments import validate_args as validate_args_original

def core_transformer_config_from_args(args, config_class=None):
    """
    config_class,用于添加model_config
    """
    config_class = config_class or TransformerConfig

    # Translate args to core transformer configuration
    kw_args = {}
    for f in dataclasses.fields(config_class):
        if hasattr(args, f.name):
            kw_args[f.name] = getattr(args, f.name)
    kw_args['persist_layer_norm'] = not args.no_persist_layer_norm
    kw_args['layernorm_zero_centered_gamma'] = args.apply_layernorm_1p
    kw_args['layernorm_epsilon'] = args.norm_epsilon
    kw_args['deallocate_pipeline_outputs'] = True
    kw_args['pipeline_dtype'] = args.params_dtype
    kw_args['batch_p2p_comm'] = not args.overlap_p2p_comm
    kw_args['num_moe_experts'] = args.num_experts
    kw_args['rotary_interleaved'] = args.rotary_interleaved
    kw_args['first_pipeline_num_layers'] = args.decoder_first_pipeline_num_layers
    kw_args['last_pipeline_num_layers'] = args.decoder_last_pipeline_num_layers
    if args.swiglu:
        kw_args['activation_func'] = F.silu
        kw_args['gated_linear_unit'] = True
        kw_args['bias_activation_fusion'] = args.bias_swiglu_fusion
    else:
        kw_args['bias_activation_fusion'] = args.bias_gelu_fusion

    if args.init_method_xavier_uniform:
        kw_args['init_method'] = torch.nn.init.xavier_uniform_
        kw_args['scaled_init_method'] = torch.nn.init.xavier_uniform_
    if args.group_query_attention:
        kw_args['num_query_groups'] = args.num_query_groups
    else:
        kw_args['num_query_groups'] = None
    kw_args['config_logger_dir'] = args.config_logger_dir

    # Return config.
    return config_class(**kw_args)

def patch_if_not_exist(
        group_or_parser: Union[argparse._ArgumentGroup, argparse.ArgumentParser],
        keyname, type=None, default=None, help=None
):
    """
    检测patch中添加参数是否存在，如果不存在才进行添加
    """
    has_keyname = False
    for action in vars(group_or_parser)["_actions"]:
        if isinstance(action, argparse._StoreAction):
            if keyname in action.option_strings:
                has_keyname = True

    if not has_keyname:
        return group_or_parser.add_argument(
            keyname,
            type=type,
            default=default,
            help=help,
        )
    return None

def patch_choices_if_exist(
    parser: argparse.ArgumentParser,
    keyname: str,
    new_choices: Union[str, List[str]]
) -> None:
    """
    查找一个已存在的参数，并为其 'choices' 列表添加新的选项。

    Args:
        parser: argparse.ArgumentParser 对象。
        keyname: 要查找的参数名 (例如 '--dataloader-type')。
        new_choices: 要添加的一个或多个新选项。
    """
    if isinstance(new_choices, str):
        new_choices = [new_choices]

    # 遍历所有 action (包括在 group 中) 来查找参数
    for action in parser._actions:
        if keyname in action.option_strings:
            if hasattr(action, 'choices') and action.choices is not None:
                for choice in new_choices:
                    if choice not in action.choices:
                        action.choices.append(choice)
                print(f"Patch Info: Extended '{keyname}' choices to: {action.choices}")
            else:
                print(f"Warning: Argument '{keyname}' found but it does not have 'choices'.")
            return # 找到并处理后即可返回

    print(f"Warning: Argument '{keyname}' not found to patch its choices.")

def get_patch_args(parser):
    """
    对patch参数进行解析添加
    """
    group = parser.add_argument_group(title="patch")

    for action in vars(group)["_actions"]:
        if isinstance(action, argparse._StoreAction):
            if "--tokenizer-type" in action.option_strings:
                action.default = "NullTokenizer"

    for action in vars(group)["_actions"]:
        if isinstance(action, argparse._StoreAction):
            if "--vocab-size" in action.option_strings:
                action.default = -1

    for action in vars(group)["_actions"]:
        if isinstance(action, argparse._StoreAction):
            if "--optimizer" in action.option_strings:
                action.choices.append("hybridadam")

    for action in vars(group)["_actions"]:
        if isinstance(action, argparse._StoreAction):
            if "--position-embedding-type" in action.option_strings:
                action.choices.append("none")

    has_rotary_base = False
    for action in vars(group)["_actions"]:
        if isinstance(action, argparse._StoreAction):
            if "--rotary-base" in action.option_strings:
                has_rotary_base = True

    patch_if_not_exist(
        group,
        "--rotary-base",
        type=int,
        default=10000,
        help="Base to use for rotary positional embeddings, default 10000",
    )

    patch_if_not_exist(
        group,
        "--local-rank",
        type=int,
        default=None,
        help="local rank passed from distributed launcher",
    )

    patch_if_not_exist(
        group,
        "--spatial-merge-size",
        type=int,
        default=2,
    )

    patch_if_not_exist(
        group,
        "--temporal-patch-size",
        type=int,
        default=2,
    )

    patch_if_not_exist(
        group,
        "--patch-size",
        type=int,
        default=14,
    )

    group.add_argument("--n-head-kv", type=int, default=None, help="n-head-kv")

    group.add_argument(
        "--transformer-type", type=str, default="megatron", help="transformer-type"
    )

    group.add_argument(
        "--max-padding-length", type=int, default=None, help="max-padding-length"
    )

    group.add_argument("--dataset", type=str, default=None, help="dataset")

    group.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of finetunning epochs. Zero results in " "evaluation only.",
    )

    group.add_argument(
        "--intermediate-size", type=int, default=None, help="--intermediate-size"
    )

    group.add_argument(
        "--extra-vocab-size", type=int, default=0, help="--extra-vocab-size"
    )

    group.add_argument(
        "--keep-last",
        action="store_true",
        help="Keep the last batch (maybe incomplete) in" "the data loader",
    )

    group.add_argument("--data-dir", default=None, help="data-dir")

    group.add_argument(
        "--train-data",
        nargs="+",
        default=None,
        help="Whitespace separated paths or corpora names " "for training.",
    )

    group.add_argument(
        "--valid-data", nargs="+", default=None, help="path(s) to the validation data."
    )

    group.add_argument("--patch-tokenizer-type", type=str, help="patch-tokenizer-type")

    group.add_argument(
        "--use-alibi-mask",
        action="store_true",
        help="use alibi mask for baichuan model",
    )

    group.add_argument("--use-normhead", action="store_true", help="use-normhead")

    group.add_argument("--glu-activation", type=str, help="GLU activations to use.")

    group.add_argument(
        "--attention-head-type",
        type=str,
        default=None,
        choices=["multihead", "multiquery"],
        help="Type of attention heads. `multihead` is the standard multi-head attention."
        "`multiquery` shares the values and keys across attention heads",
    )

    group.add_argument(
        "--transformer-timers",
        action="store_true",
        help="If set, activate the timers within the transformer layers."
        "Only for debugging, as this slows down the model.",
    )

    group.add_argument("--text-generate-input-file", type=str, default="")

    group.add_argument("--text-generate-output-file", type=str, default="")

    group.add_argument("--text-generate-gt-file", type=str, default="")

    group.add_argument(
        "--time",
        action="store_true",
        help="measure end to end text generation average time",
    )

    group.add_argument("--eval-dev", action="store_true")

    group.add_argument(
        "--input-len",
        type=int,
        default=1,
        help="input lenth for measure end to end text generation average time",
    )

    group.add_argument(
        "--generation-length", type=int, default=None, help="generation-seq-len"
    )

    group.add_argument("--top-p", type=float, default=0.0, help="Top p sampling.")

    group.add_argument("--top-k", type=int, default=0, help="Top k sampling.")

    group.add_argument(
        "--out-seq-length",
        type=int,
        default=1024,
        help="Size of the output generated text.",
    )

    group.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature."
    )

    group.add_argument(
        "--repetition_penalty", type=float, default=1.1, help="Repetition_penalty."
    )

    group.add_argument(
        "--embed-layernorm", action="store_true", help="use layernorm for embedding"
    )

    group.add_argument(
        "--repetition-penalty", type=float, default=1.2, help="Repetition_penalty."
    )

    group.add_argument(
        "--source-seq-len", type=int, default=None, help="source-seq-len"
    )

    group.add_argument(
        "--target-seq-len", type=int, default=None, help="target-seq-len"
    )

    group.add_argument(
        "--position-encoding-2d", action="store_true", help="position-encoding-2d"
    )

    group.add_argument(
        "--z-loss-weight",
        type=float,
        default=0.0,
        help="the max-z weight for baichuan2",
    )

    group.add_argument(
        "--use-llama2-rotary-position-embeddings",
        action="store_true",
        help="Use llama2 rotary positional embeddings or not. "
        "Deprecated: use --position-embedding-type",
    )

    group.add_argument(
        "--use-mistral-rotary-position-embeddings",
        action="store_true",
        help="Use llama2 rotary positional embeddings or not. "
        "Deprecated: use --position-embedding-type",
    )

    group.add_argument("--mm-use-im-start-end", action="store_true")

    group.add_argument("--mm-use-im-patch-token", action="store_true")

    group.add_argument("--tune-mm-mlp-adapter", action="store_true")

    group.add_argument("--freeze-clip-vision-tower", action="store_true")

    group.add_argument("--freeze-llm", action="store_true")

    group.add_argument("--image-folder", type=str, default="")

    group.add_argument("--mm-vision-select-layer", type=int, default=None)

    group.add_argument("--vision-tower", type=str, default="")

    group.add_argument("--image-aspect-ratio", type=str, default="square")

    group.add_argument("--version", type=str, default="plain")

    group.add_argument("--mm-projector-type", type=str, default=None)

    group.add_argument("--image-size", type=int, default=None, help="image-size")

    group.add_argument("--rotary-scale-factor", type=int, default=1)

    group.add_argument("--cvcuda-image-processing", action="store_true")

    group.add_argument(
        "--expert-tensor-parallelism",
        action="store_true",
        default=False,
        help="use tensor parallelism for expert layers in MoE",
    )

    group.add_argument(
        "--expert-interval",
        type=int,
        default=2,
        help='Use experts in every "expert-interval" layers',
    )

    group.add_argument("--moe", action="store_true")

    group.add_argument("--moe-topk", type=int, default=1, help="moe-topk")

    group.add_argument(
        "--moe-expert-parallel-size",
        type=int,
        default=None,
        help="Degree of the MoE expert parallelism. By default, "
        "the size of this value will be automatically determined.",
    )

    group.add_argument(
        "--moe-train-capacity-factor",
        type=float,
        default=1.0,
        help="The capacity of the MoE expert at training time",
    )

    group.add_argument(
        "--moe-eval-capacity-factor",
        type=float,
        default=1.0,
        help="The capacity of the MoE expert at eval time.",
    )

    group.add_argument(
        "--moe-min-capacity",
        type=int,
        default=4,
        help="The minimum capacity per MoE expert regardless of the capacity_factor.",
    )

    group.add_argument(
        "--moe-loss-coeff",
        type=float,
        default=0.01,
        help="Scaling coefficient for adding MoE loss to model loss",
    )

    group.add_argument(
        "--use-tutel", action="store_true", help="Use Tutel optimization for MoE"
    )

    group.add_argument(
        "--router-type",
        type=str,
        default="topk",
        choices=["topk", "expert_choice"],
        help="Options for router type, support top1 & top2 and expert_choice",
    )

    group.add_argument(
        "--moe-input-feature-slicing",
        action="store_true",
        help="Enable moe all2all performance optimization.",
    )

    group.add_argument(
        "--disable-bias-linear-fc",
        action="store_false",
        help="Disable bias in the linear layers",
        dest="add_bias_linear_fc",
    )

    group.add_argument(
        "--disable-bias-attn-fc",
        action="store_false",
        help="Disable bias in the linear layers",
        dest="add_bias_attn_fc",
    )

    group.add_argument(
        "--disable-parallel-output",
        action="store_false",
        help="Disable parallel-output",
        dest="enable_parallel_output",
    )

    group.add_argument(
        "--task-list",
        type=str,
        default="all",
        help='Either "all" or comma separated list of tasks.',
    )

    group.add_argument(
        "--verbosity",
        type=str,
        default="INFO",
        help="Logging verbosity",
    )

    group.add_argument(
        "--adaptive-seq-len",
        default=False,
        action="store_true",
        help="Should the sequence length be adapted to the batch during evaluation,"
        " if in fp16 the results will be slightly different due to numerical"
        " errors but greatly speed up evaluation.",
    )

    group.add_argument(
        "--eval-fp32",
        default=False,
        action="store_true",
        help="Should the evaluation run in fp32",
    )

    group.add_argument("--num-fewshot", type=int, default=None, help="num fewshot")

    group.add_argument(
        "--convert-checkpoint-from-megatron-to-transformers",
        action="store_true",
        help=(
            "If True, convert a Megatron checkpoint to a Transformers checkpoint. "
            "If False, convert a Transformers checkpoint to a Megatron checkpoint."
        ),
    )

    group.add_argument("--shared-moe-ffn-hidden-size", type=int, default=None)

    group.add_argument(
        "--enable-shared-expert", action="store_true", help="enable-shared-expert"
    )

    patch_if_not_exist(
        group,
        "--moe-ffn-hidden-size", type=int, default=None
    )

    patch_if_not_exist(
        group,
        "--q-lora-rank", type=int, default=None
    )

    patch_if_not_exist(
        group,
        "--kv-lora-rank", type=int, default=None
    )

    patch_if_not_exist(
        group,
        "--v-head-dim", type=int, default=None
    )

    group.add_argument("--qk-nope-head-dim", type=int, default=None)
    group.add_argument("--qk-rope-head-dim", type=int, default=None)
    group.add_argument("--num-shared-experts", type=int, default=None)
    
    patch_if_not_exist(
        group,
        "--moe-layer-freq", type=int, default=1 
    )

    patch_if_not_exist(
        group,
        "--rotary-scaling-factor", type=int, default=1
    )

    group.add_argument(
        "--optimizer-offload-policy",
        default="static",
        type=str,
        help="Optimizer Offload Policy used by OffloadDistributedOptimizer, "
        "valid if base optimizer is HybridAdam.",
    )

    patch_if_not_exist(
        group,
        "--optimizer-offload-fraction", type=float, default=0.5 
    )

    group.add_argument(
        "--train-mode", default="pretrain", type=str, help="pretrain or finetune"
    )

    group.add_argument(
        "--optimizer-offload-auto-threshold",
        type=int,
        default=2048 * 1024 * 1024,
        help="Optimizer Offload Threshold currently used by auto policy, "
        "tune larger if OOM occurs",
    )

    group.add_argument(
        "--optimizer-offload-chunk-size",
        type=int,
        default=32 * 1024 * 1024,
        help="Chunk size of Chunk Manager in Optimizer Offload,"
        "keep zero to search for a optimal size",
    )

    group.add_argument(
        "--cpu-offloading",
        default=False,
        action="store_true",
        help="Use activation checkpointing.",
    )

    group.add_argument(
        "--cpu-offloading-num-layers",
        type=int,
        default=0,
        help="The num of layers to be moved to CPU",
    )

    group.add_argument('--dataset-config', type=str, default=None)
    group.add_argument("--prompt-path", type=str, default=None)
    group.add_argument('--freeze-LM', action='store_true', default=False)
    group.add_argument('--freeze-ViT', action='store_true', default=False)
    group.add_argument('--language-model-type', type=str, required=False)
    group.add_argument('--vision-model-type', type=str, default="clip")
    group.add_argument("--disable-vision-class-token", action="store_true", default=False)
    group.add_argument(
        "--allow-missing-vision-projection-checkpoint", action="store_true", default=False
    )
    group.add_argument("--use-te", action="store_true", default=False)
    group.add_argument(
        "--dataloader-save", type=str, default=None, help="Energon dataloader state save path"
    )
    group.add_argument(
        "--use-tiling", action="store_true", default=False, help="Use input image tiling"
    )
    group.add_argument("--max-num-tiles", type=int, default=1, help="Maximum number of image tiles")
    group.add_argument(
        "--use-thumbnail", action="store_true", default=False, help="Add image thumbnail as a tile"
    )
    group.add_argument(
        "--dataloader-seq-length",
        type=int,
        help="Make dataloader to produce sequences of specific length.",
    )
    group.add_argument(
        "--num-frames",
        type=int,
        default=1,
        help="Number of frames to regularly sample from the video as input to the model.",
    )
    group.add_argument(
        "--online-evaluation-config", type=str, help="Config file for online evaluation."
    )

    group.add_argument(
        "--tokenizer-prompt-format",
        type=str,
        choices=["mistral", "llama3", "chatml"],
        required=False,
        help="Prompt format to use with the tokenizer.",
    )

    group.add_argument(
        "--special-tokens",
        nargs="*",
        default=["<image>"],
        help="Special tokens used in the multimodal model",
    )

    group.add_argument(
        "--image-tag-type",
        type=str,
        choices=["nvlm", "internvl", ""],
        default="",  # Default: Image tag not used.
        help="Surround image tokens with tags.",
    )

    group.add_argument(
        "--merge-in-order",
        default=False,
        action="store_true",
        help="是否顺序拼接数据集",
    )

    group.add_argument("--port", type=int, default=5000,
                       help='port for text generation server to run on')

    group.add_argument("--infer-input-file", type=str, default="",
                       help='推理的输入文件')
    
    group.add_argument("--infer-output-file", type=str, default="",
                       help='推理的输出文件')

    # cyq:如下配置用于加权的数据配比
    group.add_argument(
        "--weighted-config-file",
        type=str,
        help="Config file for weighted dataset sampling.",
    )
    group.add_argument(         
        "--domain-weights",
        type=float,
        nargs='*',
    )
    group.add_argument(         
        "--consumed-per-dataset",
        type=int,
        nargs='*',
    )
    group.add_argument(
        "--domain-names",
        type=str,
        nargs='*',
    )
    group.add_argument(
        "--weighted-train-data-paths",
        type=str,
        nargs='*',
    )
    group.add_argument(
        "--weighted-validate-data-paths",
        type=str,
        nargs='*',
    )
    group.add_argument(         
        "--budget-k",
        type=int,
    )
    group.add_argument(         
        "--ref-loss",
        type=float,
        nargs='*',
    )
    # 蒸馏 & mtp 相关
    group.add_argument(
        '--use-distillation', 
        action='store_true',
        help='Enable distillation loss.'
    )
    group.add_argument(
        "--teacher-type",
        type=str,
        choices=["real", "fake", "none", "local"],
        default="none",
    )
    group.add_argument(
        "--teacher-ip",
        type=str,
    )
    group.add_argument(
        "--teacher-port",
        type=int,
    )
    group.add_argument(
        '--send-dataset-name', 
        type=bool,
        default=True,
        help='Dump with dataset name.'
    )
    group.add_argument(
        '--distill-loss-weight', 
        type=float, default=0.7,
        help='Weight for the distillation loss component.'
    )
    group.add_argument(
        '--jsd-loss-weight', 
        type=float, default=0.0,
        help='Weight for the JSD loss component.'
    )
    group.add_argument(
        '--jsd-beta', 
        type=float, default=0.5,
        help='interpolation factor of JSD loss.'
    )
    group.add_argument(
        '--lm-loss-weight', 
        type=float, default=0.3,
        help='Weight for the language model loss component when distillation is enabled.'
    )
    group.add_argument(
        '--mtp-loss-weight', 
        type=float, default=0.1,
        help='Weight for the MTP loss.'
        )
    group.add_argument(
        '--mtp-distill-loss-weight', 
        type=float, default=0.1,
        help='Weight for the MTP distillation loss.'
    )
    group.add_argument(
        '--mtp-jsd-loss-weight', 
        type=float, default=0.0,
        help='Weight for the MTP JSD loss.'
    )
    group.add_argument(
        '--mtp-jsd-beta', 
        type=float, default=0.5,
        help='interpolation factor of MTP JSD loss.'
    )
    group.add_argument(
        '--monitor-prob-diff',
        action='store_true',
        default=False,
        help='Enable teacher/student probability difference monitoring. '
             'Writes distill/prob_diff_top, distill/prob_diff_bot and distill/prob_leak to TensorBoard.'
    )
    group.add_argument(
        '--sliding-window', 
        type=int,
        default=None
    )
    group.add_argument(
        '--use-every-k-full-attention',
        action='store_true'
    )
    group.add_argument(
        '--num-k-full-attn',
        type=int,
    )
    return parser

def validate_args_patched(args, defaults={}):
    # 首先调用原生的参数验证函数
    args = validate_args_original(args, defaults)

    if args.use_distillation and args.teacher_type == "real":
        assert args.teacher_ip is not None and args.teacher_port is not None, "蒸馏时需要提供teacher ip port"

    if args.weighted_config_file:
        assert args.weighted_config_file.endswith(".json"), "数据采样配置文件需要是json文件"
        weighted_config = json.load(open(args.weighted_config_file))
        args.domain_weights, args.consumed_per_dataset, args.domain_names = [], [], []
        args.weighted_train_data_paths, args.ref_loss = [], []
        args.weighted_train_data_file_sizes = []  # 预存文件大小，避免启动时重复 stat
        # 用于检查点保存，始终存储 training 中原始的 val_file_path
        args.original_weighted_validate_data_paths = []
        for domain_item in weighted_config["training"]:
            # args.weighted_train_data_paths.append(domain_item["file_path"])
            args.weighted_train_data_paths.append({
                "file_path": domain_item["file_path"],
                "quality_id": domain_item.get("quality_id", 0),
            })
            args.weighted_train_data_file_sizes.append(domain_item.get("file_size"))  # 可能不存在
            # 存储原始的 val_file_path 以便保存检查点
            args.original_weighted_validate_data_paths.append(domain_item["val_file_path"])
            args.domain_names.append(domain_item["name"])
            args.domain_weights.append(domain_item["weight"])
            args.consumed_per_dataset.append(domain_item["consumed_data_count"])
            args.ref_loss.append(domain_item["ref_loss"])
        # -- 处理 "validation" 列表 (如果存在) --
        validation_data = weighted_config.get("validation")
        if validation_data and isinstance(validation_data, list):
            print("Patch Info: Found 'validation' list in config, using it for evaluation.")
            args.has_separate_validation_config = True
            args.weighted_validate_data_paths = [item["val_file_path"] for item in validation_data]
            args.weighted_validate_domain_names = [item["name"] for item in validation_data]
        else:
            print("Patch Info: 'validation' list not found, falling back to 'val_file_path' from 'training' list.")
            args.has_separate_validation_config = False
            args.weighted_validate_data_paths = args.original_weighted_validate_data_paths
            args.weighted_validate_domain_names = args.domain_names
    return args