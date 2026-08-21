"""Fused tensor-parallel cross entropy and distillation losses.

The module returns three per-token losses:

* language-model cross entropy;
* forward KL, ``KL(teacher_topk || student_full_vocab)``;
* optional reverse KL, ``KL(student_topk || teacher_topk)``.

Only the student logits receive gradients. Teacher logits and indices are treated
as distillation targets.
"""

import torch
import torch.nn.functional as F

from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from megatron.core.tensor_parallel.utils import VocabUtility


def calc_student_topk_logits(
    student_vocab_parallel_logits,
    teacher_topk_indices,
    vocab_start_index,
    vocab_end_index,
):
    """Gather the teacher top-k entries owned by the current TP rank."""
    valid_mask = (teacher_topk_indices >= vocab_start_index) & (
        teacher_topk_indices < vocab_end_index
    )
    index_offset = torch.where(
        valid_mask,
        teacher_topk_indices - vocab_start_index,
        torch.zeros_like(teacher_topk_indices),
    ).to(torch.int64)

    student_topk_logits = torch.gather(
        student_vocab_parallel_logits,
        dim=-1,
        index=index_offset,
    )
    student_topk_logits = student_topk_logits * valid_mask.to(
        student_topk_logits.dtype
    )
    return valid_mask, index_offset, student_topk_logits


def calc_distill_loss_and_softmax(
    student_topk_logits,
    student_softmax,
    student_logits_max,
    student_exp_logits_sum,
    teacher_topk_logits,
):
    """Calculate forward KL and normalize the local student probabilities."""
    student_topk_log_probs = (
        student_topk_logits
        - student_logits_max.unsqueeze(dim=-1)
        - torch.log(student_exp_logits_sum).unsqueeze(dim=-1)
    )
    student_softmax.div_(student_exp_logits_sum.unsqueeze(dim=-1))

    teacher_topk_softmax = torch.softmax(teacher_topk_logits, dim=-1)
    loss = F.kl_div(
        student_topk_log_probs,
        teacher_topk_softmax,
        reduction="none",
    ).sum(dim=-1)
    return loss, student_softmax, teacher_topk_softmax


def get_tp_vocab_logits(
    vocab_parallel_logits,
    target,
    vocab_start_index,
    vocab_end_index,
):
    """Return local target logits and the masks needed by backward."""
    target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
    masked_target = target - vocab_start_index
    masked_target = masked_target.masked_fill(target_mask, 0).to(torch.int64)

    partition_vocab_size = vocab_parallel_logits.size(-1)
    logits_2d = vocab_parallel_logits.reshape(-1, partition_vocab_size)
    masked_target_1d = masked_target.reshape(-1)
    row_indices = torch.arange(logits_2d.size(0), device=logits_2d.device)

    predicted_logits = logits_2d[row_indices, masked_target_1d]
    predicted_logits = predicted_logits.reshape_as(target).clone().contiguous()
    predicted_logits.masked_fill_(target_mask, 0.0)
    return predicted_logits, target_mask, masked_target_1d


class _VocabDistillParallelCrossEntropyV2(torch.autograd.Function):
    """Autograd implementation for tensor-parallel distillation losses."""

    @staticmethod
    def forward(
        ctx,
        student_vocab_parallel_logits,
        labels,
        teacher_topk_logits,
        teacher_topk_indices,
        use_reverse_kl,
    ):
        if student_vocab_parallel_logits.shape[:-1] != labels.shape:
            raise ValueError(
                "labels must match student logits except for the vocabulary dimension"
            )
        if teacher_topk_logits.shape != teacher_topk_indices.shape:
            raise ValueError("teacher_topk_logits and teacher_topk_indices must match")
        if teacher_topk_logits.shape[:-1] != labels.shape:
            raise ValueError(
                "teacher top-k tensors must have the same token shape as labels"
            )
        if teacher_topk_logits.size(-1) == 0:
            raise ValueError("teacher top-k tensors must contain at least one entry")

        partition_vocab_size = student_vocab_parallel_logits.size(-1)
        rank = get_tensor_model_parallel_rank()
        world_size = get_tensor_model_parallel_world_size()
        vocab_start_index, vocab_end_index = (
            VocabUtility.vocab_range_from_per_partition_vocab_size(
                partition_vocab_size,
                rank,
                world_size,
            )
        )
        tp_group = get_tensor_model_parallel_group()

        teacher_valid_mask, teacher_index_offset, student_topk_logits = (
            calc_student_topk_logits(
                student_vocab_parallel_logits,
                teacher_topk_indices,
                vocab_start_index,
                vocab_end_index,
            )
        )
        torch.distributed.all_reduce(
            student_topk_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=tp_group,
        )

        student_logits_max = student_vocab_parallel_logits.max(dim=-1).values
        torch.distributed.all_reduce(
            student_logits_max,
            op=torch.distributed.ReduceOp.MAX,
            group=tp_group,
        )

        # Do not overwrite the input tensor. This keeps the custom operation safe
        # when callers reuse the logits or pass a leaf tensor during testing.
        shifted_logits = student_vocab_parallel_logits - student_logits_max.unsqueeze(-1)
        predicted_logits, target_mask, local_label_index = get_tp_vocab_logits(
            shifted_logits,
            labels,
            vocab_start_index,
            vocab_end_index,
        )

        student_softmax = torch.exp(shifted_logits)
        student_exp_logits_sum = student_softmax.sum(dim=-1)

        torch.distributed.all_reduce(
            predicted_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=tp_group,
        )
        torch.distributed.all_reduce(
            student_exp_logits_sum,
            op=torch.distributed.ReduceOp.SUM,
            group=tp_group,
        )

        loss = torch.log(student_exp_logits_sum) - predicted_logits
        distill_loss, student_softmax, teacher_topk_softmax = (
            calc_distill_loss_and_softmax(
                student_topk_logits,
                student_softmax,
                student_logits_max,
                student_exp_logits_sum,
                teacher_topk_logits,
            )
        )

        reverse_kl_loss = torch.zeros_like(loss)
        reverse_kl_grad_topk = torch.empty(
            0,
            device=student_vocab_parallel_logits.device,
            dtype=student_vocab_parallel_logits.dtype,
        )
        if use_reverse_kl:
            student_topk_log_softmax = F.log_softmax(student_topk_logits, dim=-1)
            student_topk_softmax = student_topk_log_softmax.exp()
            teacher_topk_log_softmax = F.log_softmax(teacher_topk_logits, dim=-1)
            log_ratio = student_topk_log_softmax - teacher_topk_log_softmax
            reverse_kl_loss = (student_topk_softmax * log_ratio).sum(dim=-1)

            # d KL(q || p) / d student_topk_logits
            reverse_kl_grad_topk = student_topk_softmax * (
                log_ratio - reverse_kl_loss.unsqueeze(dim=-1)
            )

        ctx.use_reverse_kl = bool(use_reverse_kl)
        ctx.save_for_backward(
            student_softmax,
            teacher_index_offset,
            teacher_topk_softmax,
            teacher_valid_mask,
            target_mask,
            local_label_index,
            reverse_kl_grad_topk,
        )
        return loss, distill_loss, reverse_kl_loss

    @staticmethod
    def backward(ctx, grad_output, distill_grad_output, reverse_kl_grad_output):
        (
            student_softmax,
            teacher_index_offset,
            teacher_topk_softmax,
            teacher_valid_mask,
            target_mask,
            local_label_index,
            reverse_kl_grad_topk,
        ) = ctx.saved_tensors

        partition_vocab_size = student_softmax.size(-1)
        row_count = student_softmax.numel() // partition_vocab_size
        row_indices = torch.arange(row_count, device=student_softmax.device)

        # Language-model cross-entropy gradient.
        grad_input = student_softmax.clone()
        grad_2d = grad_input.reshape(-1, partition_vocab_size)
        softmax_update = (~target_mask).reshape(-1).to(grad_2d.dtype)
        grad_2d[row_indices, local_label_index] -= softmax_update
        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        # Forward-KL gradient: q_student - p_teacher.
        distill_grad_input = student_softmax.clone()
        distill_grad_2d = distill_grad_input.reshape(-1, partition_vocab_size)
        top_k = teacher_index_offset.size(-1)
        local_teacher_probs = torch.where(
            teacher_valid_mask,
            teacher_topk_softmax,
            torch.zeros_like(teacher_topk_softmax),
        )
        distill_grad_2d.scatter_add_(
            dim=-1,
            index=teacher_index_offset.reshape(-1, top_k),
            src=-local_teacher_probs.reshape(-1, top_k),
        )
        distill_grad_input.mul_(distill_grad_output.unsqueeze(dim=-1))
        grad_input.add_(distill_grad_input)

        # Reverse-KL is defined only on the teacher's top-k support.
        if ctx.use_reverse_kl:
            local_reverse_grad = torch.where(
                teacher_valid_mask,
                reverse_kl_grad_topk,
                torch.zeros_like(reverse_kl_grad_topk),
            )
            reverse_grad_input = torch.zeros_like(student_softmax)
            reverse_grad_input.reshape(-1, partition_vocab_size).scatter_add_(
                dim=-1,
                index=teacher_index_offset.reshape(-1, top_k),
                src=local_reverse_grad.reshape(-1, top_k),
            )
            reverse_grad_input.mul_(reverse_kl_grad_output.unsqueeze(dim=-1))
            grad_input.add_(reverse_grad_input)

        return grad_input, None, None, None, None


def fused_distill_vocab_parallel_cross_entropy_v2(
    student_vocab_parallel_logits,
    labels,
    teacher_topk_logits,
    teacher_topk_indices,
    use_reverse_kl: bool = False,
):
    """Compute CE, forward-KL, and optional reverse-KL per token.

    Args:
        student_vocab_parallel_logits: ``[..., vocab_size_per_partition]``.
        labels: Global vocabulary ids with shape ``[...]``.
        teacher_topk_logits: Teacher logits with shape ``[..., top_k]``.
        teacher_topk_indices: Global vocabulary ids with shape ``[..., top_k]``.
        use_reverse_kl: Whether to calculate ``KL(student_topk || teacher_topk)``.

    Returns:
        A tuple ``(loss, distill_loss, reverse_kl_loss)``. Every item has the
        same shape as ``labels``. When reverse KL is disabled, its loss is zero.
    """
    return _VocabDistillParallelCrossEntropyV2.apply(
        student_vocab_parallel_logits,
        labels,
        teacher_topk_logits,
        teacher_topk_indices,
        use_reverse_kl,
    )
