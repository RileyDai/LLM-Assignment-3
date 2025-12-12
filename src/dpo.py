# src/dpo.py

import torch
import torch.nn.functional as F

def dpo_loss(
    logp_chosen, logp_rejected,
    logp_ref_chosen, logp_ref_rejected,
    beta=0.1
):
    """
    Direct Preference Optimization (DPO) loss.

    logp_chosen:     πθ(y+|x)
    logp_rejected:   πθ(y-|x)
    logp_ref_*:      πref(y|x)
    beta:            0.1

    loss = -log σ( β( (logπθ(y+)-logπθ(y-)) - (logπref(y+)-logπref(y-)) ) )
    """

    pi_diff      = logp_chosen - logp_rejected
    ref_diff     = logp_ref_chosen - logp_ref_rejected

    advantage    = beta * (pi_diff - ref_diff)

    loss = -F.logsigmoid(advantage).mean()

    return loss
