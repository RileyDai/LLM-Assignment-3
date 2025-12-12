# train_ppo.py
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from src.reward_model import RewardModel
from src.rl_utils import compute_seq_logprobs
from src.ppo import ppo_loss

def sample_once(policy, tokenizer, prompts, device, max_new_tokens=64):
    policy.eval()
    inputs = tokenizer(
        prompts,
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

    # ---------------------------
    # Adaptive KL parameters
    # ---------------------------
    kl_coef = 0.1
    target_kl = 1.0

    for step in range(10):  # sanity check
        batch_prompts = prompts[step % len(prompts): (step % len(prompts)) + 4]
        if len(batch_prompts) < 4:
            batch_prompts = prompts[:4]

        texts, input_ids, attn = sample_once(policy, tokenizer, batch_prompts, device)

        # ---------------------------
        # Rewards
        # ---------------------------
        with torch.no_grad():
            rewards = reward_model(input_ids, attn)  # [B]

        # ---------------------------
        # Advantage with running mean baseline
        # ---------------------------
        running_mean = alpha * running_mean + (1 - alpha) * rewards.mean().item()
        adv = rewards - running_mean

        # Advantage Normalization
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # ---------------------------
        # Logprobs from policy & ref
        # ---------------------------
        with torch.no_grad():
            logp_old = compute_seq_logprobs(ref_policy, input_ids, attn)
        logp = compute_seq_logprobs(policy, input_ids, attn)

        # ---------------------------
        # TRUE KL Implementation
        # ---------------------------
        with torch.no_grad():
            out_new = policy(input_ids, attention_mask=attn)
            out_old = ref_policy(input_ids, attention_mask=attn)

            logp_new_tokens = torch.log_softmax(out_new.logits, dim=-1)
            logp_old_tokens = torch.log_softmax(out_old.logits, dim=-1)
            p_old = torch.exp(logp_old_tokens)

            kl_token = p_old * (logp_old_tokens - logp_new_tokens)
            kl = kl_token.sum(dim=-1).mean(dim=-1)  # [B]
            kl = torch.clamp(kl, 0, 10.0)

        # ---------------------------
        # Adaptive KL Coefficient Update
        # ---------------------------
        kl_mean = kl.mean().item()
        if kl_mean > 1.5 * target_kl:
            kl_coef *= 1.5
        elif kl_mean < target_kl / 1.5:
            kl_coef /= 1.5

        # ---------------------------
        # PPO Loss
        # ---------------------------
        loss = ppo_loss(logp, logp_old, adv, clip_ratio=0.2, kl=kl, kl_coef=kl_coef)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        print(
            f"[Step {step}] reward={rewards.mean().item():.3f}, "
            f"KL={kl_mean:.3f}, kl_coef={kl_coef:.4f}, loss={loss.item():.4f}"
        )

    torch.save(policy.state_dict(), "policy_ppo.pt")
    print("Saved policy_ppo.pt")

if __name__ == "__main__":
    main()
