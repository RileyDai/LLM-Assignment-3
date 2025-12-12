# src/grpo.py
import torch

def grpo_loss(logprobs, advantages, kl=None, kl_coef=0.0):

    advantages = advantages.detach()

    pg_loss = -(logprobs * advantages).mean()

    loss = pg_loss
    if kl is not None:
        loss = loss + kl_coef * kl.mean()

    return loss
