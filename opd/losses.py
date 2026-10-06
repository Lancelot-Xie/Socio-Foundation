"""Response-only OPD objectives with per-sequence averaging.

Sampled reverse KL is a local score-function surrogate at fixed student-visited
prefixes, refreshed after each optimizer update. It is not exact sequence KL.
Forward KL uses teacher top-k PLUS the remaining probability mass as an other
bucket. It never renormalizes away tokens the student has yet to learn.
"""

import torch


# PyTorch <= 2.9.0 can launch an invalid CUDA grid when integer-indexing a
# contiguous slice larger than roughly 256 MiB.  Keep both the indexed tensor
# and the temporary full-vocabulary tensor comfortably below that boundary.
_FORWARD_KL_WORKING_SET_BYTES = 128 * 1024 * 1024


def sequence_mean(values, mask):
    return (values * mask).sum(-1) / mask.sum(-1).clamp_min(1)


def sampled_reverse_loss(student_logp, old_logp, teacher_logp, mask, strength, clip):
    advantage = teacher_logp.detach() - old_logp.detach()
    if clip > 0:
        advantage = advantage.clamp(-clip, clip)
    # Apply g AFTER any clipping; never divide by sum(g) or normalize it away.
    loss = -(sequence_mean(student_logp * advantage, mask) * strength).mean()
    metric = sequence_mean(old_logp.detach() - teacher_logp.detach(), mask)
    return loss, metric


def topk_forward_kl(student_logp, teacher_ids, teacher_logp, vocab_size):
    teacher_logp = teacher_logp.detach().float()
    teacher_probs = teacher_logp.exp()
    student_selected = student_logp.gather(-1, teacher_ids)
    kl = (teacher_probs * (teacher_logp - student_selected)).sum(-1)
    if teacher_ids.shape[-1] < vocab_size:
        q_other = (1 - teacher_probs.sum(-1)).clamp(min=0.0, max=1.0)
        # Compute the complement in log space. 1-sum(p_topk) catastrophically loses
        # tiny tail probabilities and its clamped gradient cannot recover missing modes.
        log_p_other = student_logp.scatter(-1, teacher_ids, float("-inf")).logsumexp(-1)
        kl = kl + q_other * (q_other.clamp_min(1e-30).log() - log_p_other)
    return kl


def forward_loss(student_logp, targets, mask, strength):
    active = []
    for target in targets:
        indices = target["indices"]
        if indices.numel() == 0:
            continue
        indices = indices.to(student_logp.device)
        active.append((target, indices))

    # Integer indexing copies [selected_rows, response_tokens, vocab].  Qwen3's
    # 151,936-token vocabulary crosses the buggy CUDA kernel limit at only 442
    # float32 response positions.  Chunking the response axis preserves the
    # exact objective while avoiding that launch and bounding scatter memory.
    selected_rows = max((indices.numel() for _, indices in active), default=1)
    bytes_per_position = student_logp.shape[-1] * student_logp.element_size() * selected_rows
    chunk_size = max(1, _FORWARD_KL_WORKING_SET_BYTES // max(1, bytes_per_position))
    chunks = []
    for start in range(0, student_logp.shape[-2], chunk_size):
        stop = min(start + chunk_size, student_logp.shape[-2])
        per_chunk = student_logp[:, start:stop, 0] * 0.0
        for target, indices in active:
            part = topk_forward_kl(
                student_logp[indices, start:stop],
                target["ids"][:, start:stop],
                target["log_probs"][:, start:stop],
                target["vocab_size"],
            )
            per_chunk = per_chunk.index_add(
                0,
                indices,
                part * target["alpha"].unsqueeze(-1),
            )
        chunks.append(per_chunk)

    per_token = torch.cat(chunks, dim=-1) if chunks else student_logp[..., 0] * 0.0

    per_sequence = sequence_mean(per_token, mask)
    return (per_sequence * strength).mean(), per_sequence.detach()
