import torch
from torch.functional import F

from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from megatron.core.tensor_parallel.utils import VocabUtility

# Global stats dict: written every forward pass, read by training_log to push to TensorBoard.
# Keys: 'distill/student_topk_mass', 'distill/student_non_topk_mass'
_DISTILL_STATS: dict = {}


def calc_student_topk_logits(
        student_vocab_parallel_logits, 
        teacher_topk_indices,
        vocab_start_index, 
        vocab_end_index):
    valid_mask = (teacher_topk_indices >= vocab_start_index) & (teacher_topk_indices < vocab_end_index)
    index_offset = torch.where(valid_mask, teacher_topk_indices - vocab_start_index,
                            torch.zeros_like(teacher_topk_indices)).to(torch.int64)
    student_topk_logits = torch.gather(
        student_vocab_parallel_logits,
        dim=-1,
        index=index_offset
    ) * valid_mask
    return valid_mask, index_offset, student_topk_logits


def calc_distill_loss_and_softmax(student_topk_logits, student_softmax,
    student_exp_logits_sum, teacher_topk_logits):
    """
    student_topk_logits: 已经是 (z_k - max) / T，不再重复减 max
    student_softmax:     exp((z - max) / T)，未归一化
    student_exp_logits_sum: sum_j exp((z_j - max) / T)，全局聚合后的分母
    teacher_topk_logits: teacher 的 top-k logits / T
    """
    # log q_student_k = (z_k - max)/T - log(sum_j exp((z_j - max)/T))
    student_tok_log_softmax = student_topk_logits - torch.log(student_exp_logits_sum).unsqueeze(dim=-1)
    # student topk probability (under temperature), shape: [seq_len, batch_size, top_k]
    
    # Record topk / non-topk prob mass to _DISTILL_STATS for TensorBoard logging.
    # Only rank-0 of TP writes (student_topk_softmax is already all-reduced across TP ranks
    # before entering this function, so the value is the same on all ranks).
    with torch.no_grad():
        student_topk_softmax = torch.exp(student_tok_log_softmax)
        topk_mass = student_topk_softmax.sum(dim=-1).mean()   # scalar: mean over [seq, batch]
        _DISTILL_STATS['distill/student_topk_mass'] = topk_mass.item()
        _DISTILL_STATS['distill/student_non_topk_mass'] = max(0.0, 1.0 - topk_mass.item())
    # student_softmax -> 归一化为概率
    student_softmax.div_(student_exp_logits_sum.unsqueeze(dim=-1))
    teacher_topk_softmax = torch.softmax(teacher_topk_logits, dim=-1)
    loss = torch.sum(
        F.kl_div(student_tok_log_softmax, teacher_topk_softmax, reduction="none"),
        dim=-1,
    )
    return loss, student_softmax, teacher_topk_softmax


def get_tp_vocab_logits(vocab_parallel_logits, target, vocab_start_index, vocab_end_index):
    target_mask = (target < vocab_start_index) | (target >= vocab_end_index)

    masked_target = target.clone() - vocab_start_index
    masked_target[target_mask] = 0

    partition_vocab_size = vocab_parallel_logits.size()[-1]
    logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
    masked_target_1d = masked_target.view(-1)
    arange_1d = torch.arange(start=0, end=logits_2d.size()[0], device=logits_2d.device)
    predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
    predicted_logits_1d = predicted_logits_1d.clone().contiguous()
    predicted_logits = predicted_logits_1d.view_as(target)
    predicted_logits[target_mask] = 0.0

    return predicted_logits, target_mask, masked_target_1d


class _VocabDistillParallelCrossEntropyV2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, student_vocab_parallel_logits, labels, teacher_topk_logits, teacher_topk_indices,
                use_jsd, jsd_beta, temperature):
        get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
        partition_vocab_size = student_vocab_parallel_logits.size()[-1]
        rank = get_tensor_model_parallel_rank()
        world_size = get_tensor_model_parallel_world_size()
        vocab_start_index, vocab_end_index = get_vocab_range(partition_vocab_size, rank, world_size)
        tp_group = get_tensor_model_parallel_group()

        # Step 1: 在减 max 之前，先 gather 各 rank 的 student_topk_logits（原始 logits）
        teacher_valid_mask, teacher_index_offset, student_topk_logits = calc_student_topk_logits(
            student_vocab_parallel_logits, teacher_topk_indices, vocab_start_index, vocab_end_index)
        torch.distributed.all_reduce(student_topk_logits, op=torch.distributed.ReduceOp.SUM, group=tp_group)

        # Step 2: 计算全局 max，用于数值稳定
        student_logits_max = torch.max(student_vocab_parallel_logits, dim=-1)[0]
        torch.distributed.all_reduce(student_logits_max, op=torch.distributed.ReduceOp.MAX, group=tp_group)

        # Step 2.5: 监控 student logits 绝对值大小（在减 max 之前，反映原始 logit scale）
        with torch.no_grad():
            _DISTILL_STATS['distill/student_logits_max_mean'] = student_logits_max.mean().item()
            _DISTILL_STATS['distill/student_logits_abs_mean'] = student_vocab_parallel_logits.abs().mean().item()
            _DISTILL_STATS['distill/student_logits_norm'] = student_vocab_parallel_logits.norm(p=2).item()

        # Step 3: 原地减去 max
        student_vocab_parallel_logits -= student_logits_max.unsqueeze(dim=-1)

        # Step 4: 计算 with_temperature 版本（减完 max 后再除温度，保证数值稳定）
        student_vocab_parallel_logits_with_temperature = student_vocab_parallel_logits / temperature

        # Step 5: student_topk_logits 也减去 max 再除温度
        # shape: [seq, batch, top_k]，student_logits_max: [seq, batch] -> unsqueeze -> [seq, batch, 1]
        student_topk_logits_with_temperature = (
            student_topk_logits - student_logits_max.unsqueeze(dim=-1)
        ) / temperature

        # teacher topk logits 除以温度
        teacher_topk_logits_with_temperature = teacher_topk_logits / temperature

        # Step 6: 计算 predict_label_logits，用于 lm loss
        predict_label_logits, target_mask, local_label_index = get_tp_vocab_logits(
            student_vocab_parallel_logits, labels, vocab_start_index, vocab_end_index)

        # Step 7: 计算普通版本 softmax 和 exp_logits_sum（用于 lm loss 和 backward）
        student_softmax = student_vocab_parallel_logits
        torch.exp(student_vocab_parallel_logits, out=student_softmax)
        student_exp_logits_sum = torch.sum(student_softmax, dim=-1)

        # Step 8: 计算 with_temperature 版本 softmax 和 exp_logits_sum（用于 distill loss）
        student_softmax_with_temperature = student_vocab_parallel_logits_with_temperature.clone()
        torch.exp(student_vocab_parallel_logits_with_temperature, out=student_softmax_with_temperature)
        student_exp_logits_sum_with_temperature = torch.sum(student_softmax_with_temperature, dim=-1)

        # Step 9: 全局聚合
        torch.distributed.all_reduce(predict_label_logits, op=torch.distributed.ReduceOp.SUM, group=tp_group)
        torch.distributed.all_reduce(student_exp_logits_sum, op=torch.distributed.ReduceOp.SUM, group=tp_group)
        student_softmax.div_(student_exp_logits_sum.unsqueeze(dim=-1))
        torch.distributed.all_reduce(student_exp_logits_sum_with_temperature, op=torch.distributed.ReduceOp.SUM, group=tp_group)

        # Step 9.5: 监控 student 完整词表 entropy（精确计算，各 partition 独立算 p*log(p) 后 all_reduce SUM）
        with torch.no_grad():
            log_p = torch.log(student_softmax + 1e-20)
            entropy_per_token = -(student_softmax * log_p).sum(dim=-1)   # [seq, batch]，partial entropy
            torch.distributed.all_reduce(entropy_per_token, op=torch.distributed.ReduceOp.SUM, group=tp_group)
            _DISTILL_STATS['distill/student_entropy_mean'] = entropy_per_token.mean().item()

        # Step 10: 计算 lm loss
        loss = torch.log(student_exp_logits_sum) - predict_label_logits

        # Step 11: 计算 distill loss
        # 注意：student_topk_logits_with_temperature 已经是 (z_k - max)/T，不再传 max 进去
        distill_loss, student_softmax_with_temperature, teacher_topk_softmax_with_temperature = \
            calc_distill_loss_and_softmax(
                student_topk_logits_with_temperature,
                student_softmax_with_temperature,
                student_exp_logits_sum_with_temperature,
                teacher_topk_logits_with_temperature)

        # Step 12: 计算 JSD Loss
        jsd_loss = torch.tensor(0.0, device=student_vocab_parallel_logits.device,
                                dtype=student_vocab_parallel_logits.dtype)
        ctx.use_jsd = use_jsd
        ctx.jsd_beta = jsd_beta
        ctx.temperature = temperature

        saved_student_topk_probs = None
        saved_mixture_probs = None

        if use_jsd:
            p_teacher = teacher_topk_softmax_with_temperature
            p_student = torch.softmax(student_topk_logits_with_temperature, dim=-1)

            p_mixture = jsd_beta * p_teacher + (1.0 - jsd_beta) * p_student

            log_p_mixture = torch.log(p_mixture + 1e-20)
            kl_pm = F.kl_div(log_p_mixture, p_teacher, reduction='none', log_target=False).sum(-1)
            kl_qm = F.kl_div(log_p_mixture, p_student, reduction='none', log_target=False).sum(-1)
            jsd_loss = jsd_beta * kl_pm + (1.0 - jsd_beta) * kl_qm

            saved_student_topk_probs = p_student
            saved_mixture_probs = p_mixture

        ctx.save_for_backward(
            student_softmax, student_softmax_with_temperature, teacher_index_offset,
            teacher_topk_softmax_with_temperature, teacher_valid_mask,
            target_mask, local_label_index, saved_student_topk_probs, saved_mixture_probs)

        return loss, distill_loss, jsd_loss

    @staticmethod
    def backward(ctx, grad_output, distill_grad_output, jsd_grad_output):
        use_jsd = ctx.use_jsd
        jsd_beta = ctx.jsd_beta
        temperature = ctx.temperature

        (student_softmax, student_softmax_with_temperature,
         teacher_index_offset, teacher_topk_softmax_with_temperature, teacher_valid_mask,
         target_mask, local_label_index,
         student_topk_probs, mixture_probs) = ctx.saved_tensors

        grad_input = student_softmax
        partition_vocab_size = student_softmax.size()[-1]
        grad_2d = grad_input.view(-1, partition_vocab_size)
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=grad_2d.device)
        top_k = teacher_index_offset.size()[-1]

        # --- Distill grad: dL_distill/dz_i = (1/T) * (q_student_i - q_teacher_i) ---
        # student_softmax_with_temperature 已经是归一化后的 q_student（在 calc_distill_loss_and_softmax 中完成）
        teacher_topk_softmax_with_temperature[~teacher_valid_mask] = 0
        distill_grad_input = student_softmax_with_temperature.clone()
        distill_grad_input_2d = distill_grad_input.view(-1, partition_vocab_size)
        distill_grad_input_2d[arange_1d.unsqueeze(1), teacher_index_offset.view(-1, top_k)] -= \
            teacher_topk_softmax_with_temperature.view(-1, top_k)
        distill_grad_input.mul_(distill_grad_output.unsqueeze(dim=-1))
        distill_grad_input = distill_grad_input / temperature

        # --- LM grad: dL_lm/dz_i = q_student_i - 1{i==label} ---
        softmax_update = 1.0 - target_mask.view(-1).to(grad_2d.dtype)
        grad_2d[arange_1d, local_label_index] -= softmax_update
        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        grad_input.add_(distill_grad_input)

        # --- JSD grad ---
        if use_jsd:
            eps = 1e-20
            log_Q = torch.log(student_topk_probs + eps)
            log_M = torch.log(mixture_probs + eps)

            log_ratio = log_Q - log_M
            kl_qm = torch.sum(student_topk_probs * log_ratio, dim=-1, keepdim=True)

            jsd_grad_topk = (1.0 - jsd_beta) * student_topk_probs * (log_ratio - kl_qm)
            jsd_grad_topk.mul_(jsd_grad_output.unsqueeze(dim=-1))

            jsd_grad_input = torch.zeros_like(student_softmax_with_temperature)
            jsd_grad_input_2d = jsd_grad_input.view(-1, partition_vocab_size)

            jsd_grad_topk[~teacher_valid_mask] = 0.0
            jsd_grad_input_2d.scatter_add_(
                -1, teacher_index_offset.view(-1, top_k), jsd_grad_topk.view(-1, top_k))
            jsd_grad_input = jsd_grad_input / temperature
            grad_input.add_(jsd_grad_input)

        return grad_input, None, None, None, None, None, None


def fused_distill_vocab_parallel_cross_entropy_v2(student_vocab_parallel_logits, labels,
                                                  teacher_topk_logits, teacher_topk_indices,
                                                  use_jsd: bool = False,
                                                  jsd_beta: float = 0.5,
                                                  temperature: float = 1.0):
    """
    Args:
        student_vocab_parallel_logits: [seq, batch, vocab_size_per_partition]
        labels:                        [seq, batch]
        teacher_topk_logits:           [seq, batch, top_k]
        teacher_topk_indices:          [seq, batch, top_k]
        use_jsd:    whether to use Jensen-Shannon Divergence on top-k logits
        jsd_beta:   interpolation factor, M = beta*P_teacher + (1-beta)*Q_student
        temperature: softmax temperature for distillation
    Returns:
        loss, distill_loss, jsd_loss
    """
    return _VocabDistillParallelCrossEntropyV2.apply(
        student_vocab_parallel_logits, labels, teacher_topk_logits,
        teacher_topk_indices, use_jsd, jsd_beta, temperature)