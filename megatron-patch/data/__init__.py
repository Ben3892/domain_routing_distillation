"""Data utilities for Megatron Patch.

This package provides patched dataset providers and dataloader builders for
custom features such as domain-weighted sampling.
"""

import torch

from megatron.core import mpu
from megatron.legacy.data.data_samplers import (
    build_pretraining_data_loader as build_pretraining_data_loader_original,
)
from megatron.training import get_args, print_rank_0

from megatron_patch.tokenizer import build_tokenizer, get_tokenizer

from .easy_bin_dataset import (
    BlendedDataset,
    BlendedPTTeacherDataset,
    DomainWeightedRandomSampler,
    NumSequentialSampler,
)


def build_pretraining_data_loader(dataset, consumed_samples, is_valid=False):
    """Build the original or domain-weighted pretraining dataloader."""
    args = get_args()

    if args.dataloader_type != "weighted":
        print_rank_0("Info: Falling back to original dataloader builder.")
        return build_pretraining_data_loader_original(dataset, consumed_samples)

    if dataset is None:
        return None

    if not is_valid:
        print_rank_0(
            "Patch Info: Building DomainWeightedRandomSampler for training."
        )
        pre_dataset_len = dataset.get_pre_dataset_len()
        batch_sampler = DomainWeightedRandomSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            pre_dataset_len=pre_dataset_len,
            micro_batch_size=args.micro_batch_size,
            ref_loss=args.ref_loss,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size(),
            weights=args.domain_weights,
            consumed_per_dataset=args.consumed_per_dataset,
            domain_names=args.domain_names,
        )
    else:
        print_rank_0("Patch Info: Building NumSequentialSampler for validation.")
        pre_dataset_len = dataset.get_pre_dataset_len()
        batch_sampler = NumSequentialSampler(
            total_samples=len(dataset),
            num_dataset=len(pre_dataset_len),
            pre_dataset_len=pre_dataset_len,
            micro_batch_size=args.micro_batch_size,
            global_batch_size=args.global_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size(),
        )

    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )


def build_weighted_datasets(train_val_test_num_samples):
    """Build datasets for ``PRETRAIN-WITH-WEIGHT`` mode."""
    del train_val_test_num_samples  # Dataset sizes are derived from training args.

    args = get_args()
    if get_tokenizer() is None:
        build_tokenizer(args)
    tokenizer = get_tokenizer()

    print_rank_0("Patch Info: Building BlendedDataset for weighted pre-training.")

    total_train_samples = args.train_iters * args.global_batch_size
    total_valid_samples = args.eval_iters * args.global_batch_size

    train_dataset = BlendedDataset(
        datasets_path=args.weighted_train_data_paths,
        total_samples=total_train_samples,
        tokenizer=tokenizer,
        max_padding_length=args.max_padding_length,
        is_mmap=args.mmap_bin_files,
    )
    val_dataset = BlendedDataset(
        datasets_path=args.weighted_validate_data_paths,
        total_samples=total_valid_samples,
        tokenizer=tokenizer,
        max_padding_length=args.max_padding_length,
        is_mmap=args.mmap_bin_files,
    )
    test_dataset = BlendedDataset(
        datasets_path=args.weighted_validate_data_paths,
        total_samples=total_valid_samples,
        tokenizer=tokenizer,
        max_padding_length=args.max_padding_length,
        is_mmap=args.mmap_bin_files,
    )

    if mpu.get_data_parallel_rank() == 0:
        print_rank_0("-" * 80)
        print_rank_0("Patch Info: Actual Dataset Sizes")

        train_sizes = train_dataset.get_pre_dataset_len()
        for index, size in enumerate(train_sizes):
            print_rank_0(f"  - Training   Domain {index}: {size:,} samples")
        print_rank_0(
            f"  - Total Training Samples:   {sum(train_sizes):,} samples"
        )

        val_sizes = val_dataset.get_pre_dataset_len()
        for index, size in enumerate(val_sizes):
            print_rank_0(f"  - Validation Domain {index}: {size:,} samples")
        print_rank_0(
            f"  - Total Validation Samples: {sum(val_sizes):,} samples"
        )
        print_rank_0("-" * 80)

    return train_dataset, val_dataset, test_dataset


def build_teacher_datasets(train_val_test_num_samples):
    """Build datasets for ``PT-TEACHER`` mode."""
    del train_val_test_num_samples  # Dataset sizes come from the binary files.

    args = get_args()
    print_rank_0(
        "Patch Info: Building BlendedPTTeacherDataset for weighted pre-training."
    )
    print_rank_0(f">>>>>>>>>{args.weighted_train_data_paths}")

    file_sizes = getattr(args, "weighted_train_data_file_sizes", None)
    train_dataset = BlendedPTTeacherDataset(
        args.weighted_train_data_paths,
        args.seq_length,
        file_sizes=file_sizes,
    )
    val_dataset = BlendedPTTeacherDataset(
        args.weighted_train_data_paths,
        args.seq_length,
        file_sizes=file_sizes,
    )
    test_dataset = BlendedPTTeacherDataset(
        args.weighted_train_data_paths,
        args.seq_length,
        file_sizes=file_sizes,
    )
    return train_dataset, val_dataset, test_dataset
