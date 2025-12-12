# train_grpo.py
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from src.reward_model import RewardModel
from src.rl_utils import compute_seq_logprobs
from src.grpo import grpo_loss


def sample_group(policy, tokenizer, prompts, group_size, device, max_new_tokens=64):

    policy.eval()

    group_prompts = []
    for p in prompts:
        group_prompts.extend([p] * group_size)  # B * G 条

    inputs = tokenizer(
        group_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    ).to(device)

    with torch.no_grad():
        outputs = policy.generate(
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
    ).to(device)

    return texts, tok["input_ids"], tok["attention_mask"]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = "distilgpt2"

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    policy = AutoModelForCausalLM.from_pretrained(base_model).to(device)
    ref_policy = AutoModelForCausalLM.from_pretrained(base_model).to(device)
    ref_policy.eval()

    reward_model = RewardModel(base_model_name=base_model).to(device)
    reward_model.load_state_dict(torch.load("reward_model.pt", map_location=device))
    reward_model.eval()

    optimizer = AdamW(policy.parameters(), lr=1e-5)

    ds = load_dataset("Anthropic/hh-rlhf")["train"].select(range(32))
    prompts = [ex["chosen"][:200] for ex in ds]

    running_mean = 0.0
    alpha = 0.9

    group_size = 4  
    kl_coef = 0.05   
    target_kl = 1.0  

    for step in range(10):  
        start = (step * 4) % len(prompts)
        batch_prompts = prompts[start: start + 4]
        if len(batch_prompts) < 4:
            batch_prompts = prompts[:4]
        B = len(batch_prompts)

        texts, input_ids, attn = sample_group(
            policy, tokenizer, batch_prompts, group_size, device
        )  # N = B * G

        N = input_ids.size(0)
        assert N == B * group_size

        # ---------------------------
        # Rewards
        # ---------------------------
        with torch.no_grad():
            rewards = reward_model(input_ids, attn)  # [N]

        rewards_group = rewards.view(B, group_size)            # [B, G]
        group_mean = rewards_group.mean(dim=1, keepdim=True)   # [B, 1]
        advantages_group = rewards_group - group_mean          # [B, G]
        advantages = advantages_group.view(-1)                  # [N]

        running_mean = alpha * running_mean + (1 - alpha) * rewards.mean().item()

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ---------------------------
        # Logprobs from current policy
        # ---------------------------
        logp = compute_seq_logprobs(policy, input_ids, attn)  # [N]

        # ---------------------------
        # KL vs ref_policy（token-level KL）
        # ---------------------------
        with torch.no_grad():
            out_new = policy(input_ids, attention_mask=attn)
            out_old = ref_policy(input_ids, attention_mask=attn)

            logp_new_tokens = torch.log_softmax(out_new.logits, dim=-1)
            logp_old_tokens = torch.log_softmax(out_old.logits, dim=-1)
            p_old = torch.exp(logp_old_tokens)

            kl_token = p_old * (logp_old_tokens - logp_new_tokens)
            kl = kl_token.sum(dim=-1)        
            kl = torch.clamp(kl, 0, 10.0)

        kl_mean = kl.mean().item()
        if kl_mean > 1.5 * target_kl:
            kl_coef *= 1.5
        elif kl_mean < target_kl / 1.5:
            kl_coef /= 1.5

        # ---------------------------
        # GRPO Loss
        # ---------------------------
        loss = grpo_loss(logp, advantages, kl=kl, kl_coef=kl_coef)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        print(
            f"[Step {step}] reward_mean={rewards.mean().item():.3f}, "
            f"KL={kl_mean:.3f}, kl_coef={kl_coef:.4f}, loss={loss.item():.4f}"
        )

    torch.save(policy.state_dict(), "policy_grpo.pt")
    print("Saved policy_grpo.pt")


if __name__ == "__main__":
    main()
