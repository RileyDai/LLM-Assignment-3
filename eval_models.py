# eval_models.py
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from src.reward_model import RewardModel
from src.rl_utils import compute_seq_logprobs


def compute_kl(policy, ref_policy, input_ids, attention_mask):
    """
    Sequence-level KL:
        KL = log π_policy(y|x) - log π_ref(y|x)
    return shape [B]
    """
    with torch.no_grad():
        logp_new = compute_seq_logprobs(policy, input_ids, attention_mask)     # [B]
        logp_ref = compute_seq_logprobs(ref_policy, input_ids, attention_mask) # [B]
    return (logp_ref - logp_new)  # 这里用 KL( ref || policy )，符号无所谓，一致就行


def generate_responses(model, tokenizer, prompts, device, max_new_tokens=128):

    model.eval()

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.95,
        )

    texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    tok = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

    input_ids = tok["input_ids"].to(device)
    attn = tok["attention_mask"].to(device)

    return texts, input_ids, attn


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = "distilgpt2"

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    # reference policy
    ref_policy = AutoModelForCausalLM.from_pretrained(base_model).to(device)
    ref_policy.eval()

    # reward model
    reward_model = RewardModel(base_model_name=base_model).to(device)
    reward_model.load_state_dict(torch.load("reward_model.pt", map_location=device))
    reward_model.eval()

    def load_policy(weight_path):
        model = AutoModelForCausalLM.from_pretrained(base_model).to(device)
        state = torch.load(weight_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        return model

    models = {
        "ref":  ref_policy,
        "ppo":  load_policy("policy_ppo.pt"),
        "grpo": load_policy("policy_grpo.pt"),
        "dpo":  load_policy("policy_dpo.pt"),
    }

    # eval prompts
    raw = load_dataset("Anthropic/hh-rlhf")["train"].select(range(128))
    eval_prompts = [ex["chosen"][:200] for ex in raw]

    results = {}

    for name, policy in models.items():
        print(f"Evaluating {name} ...")

        texts, input_ids, attn = generate_responses(
            policy, tokenizer, eval_prompts, device
        )

        with torch.no_grad():
            rewards = reward_model(input_ids, attn)  # [N]

        all_kl = []
        bs = 8
        for i in range(0, input_ids.size(0), bs):
            kl_chunk = compute_kl(
                policy,
                ref_policy,
                input_ids[i:i+bs],
                attn[i:i+bs],
            )
            all_kl.append(kl_chunk.cpu())
        kl = torch.cat(all_kl, dim=0)  # [N]

        results[name] = {
            "mean_reward": float(rewards.mean().item()),
            "std_reward": float(rewards.std().item()),
            "mean_kl": float(kl.mean().item()),
            "std_kl": float(kl.std().item()),
            "num_samples": len(texts),
            "rewards": [float(x) for x in rewards.cpu()],
            "kls": [float(x) for x in kl.cpu()],
        }

        print(
            f"{name}: reward={results[name]['mean_reward']:.3f}±{results[name]['std_reward']:.3f}, "
            f"KL={results[name]['mean_kl']:.3f}±{results[name]['std_kl']:.3f}"
        )

    out_path = Path("eval_results.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved eval results to {out_path}")


if __name__ == "__main__":
    main()
