"""Training losses and schedules over padded option logits, shared by the models. `config` is a
model's own train config; only the fields named here are read."""

import math

import torch

from jev_model.metrics import option_mask, soft_cross_entropy


def lr_factor(step: int, config) -> float:
    """Linear warmup over `config.warmup` steps, then a cosine down to 0 at `config.steps`."""
    if step < config.warmup:
        return (step + 1) / config.warmup
    progress = (step - config.warmup) / max(1, config.steps - config.warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def proper_score_reward(q: torch.Tensor, labels: torch.Tensor, spherical_weight: float) -> torch.Tensor:
    """S_log + w * S_sph. The log score is floored at log(1e-4) = -9.21 so a single miss can't
    dominate the group's advantage."""
    log_score = (labels * q.clamp_min(1e-4).log()).sum(dim=-1)
    spherical = (labels * q).sum(dim=-1) / q.norm(dim=-1).clamp_min(1e-12)
    return log_score + spherical_weight * spherical


def rl_loss(logits: torch.Tensor, labels: torch.Tensor, num_options: torch.Tensor, step: int, config) -> torch.Tensor:
    """Gaussian policy over the logits (GRPO-style): perturb them `group` times, score each
    perturbed distribution with a proper scoring rule, and push the logits towards the
    perturbations that scored above the group's mean."""
    sigma = config.sigma_start + (config.sigma_end - config.sigma_start) * step / max(1, config.steps - 1)
    mask = option_mask(num_options, logits.shape[-1])
    z = logits.masked_fill(mask, 0.0)
    count = num_options.float().unsqueeze(-1)
    noise = torch.randn((config.group, *z.shape), device=z.device) * sigma
    noise = noise.masked_fill(mask, 0.0)
    # centred over the real options: a shift of every logit changes nothing after the softmax
    noise = (noise - noise.sum(dim=-1, keepdim=True) / count).masked_fill(mask, 0.0)
    sampled = z.detach() + noise
    q = sampled.masked_fill(mask, float("-inf")).softmax(dim=-1)
    reward = proper_score_reward(q, labels, config.spherical_weight)
    advantage = (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6)
    log_prob = -((sampled - z) ** 2).sum(dim=-1) / (2 * sigma**2)
    loss = -(advantage.detach() * log_prob).mean()
    if config.ce_weight:
        loss = loss + config.ce_weight * soft_cross_entropy(logits, labels, num_options).mean()
    return loss
