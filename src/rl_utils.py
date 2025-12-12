# src/rl_utils.py

import torch

def compute_seq_logprobs(model, input_ids, attention_mask):

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # [B, T, V]

    logits = logits[:, :-1, :]               # [B, T-1, V]
    target = input_ids[:, 1:]                # [B, T-1]

    log_probs = torch.log_softmax(logits, dim=-1)  # [B, T-1, V]

    token_logprobs = log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1)  # [B, T-1]

    seq_logprobs = token_logprobs.sum(dim=-1)  # [B]

    return seq_logprobs
