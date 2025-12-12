# src/ppo_core.py
import torch
import torch.nn.functional as F

def ppo_loss(logprobs, logprobs_old, advantages, clip_ratio=0.2, kl=None, kl_coef=0.0):
    ratio = torch.exp(logprobs - logprobs_old)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * advantages
    policy_loss = -torch.mean(torch.min(unclipped, clipped))
    loss = policy_loss
    if kl is not None:
        loss += kl_coef * kl.mean()
    return loss
