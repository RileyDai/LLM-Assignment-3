# train_dpo.py
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from src.rl_utils import compute_seq_logprobs
from src.dpo import dpo_loss


def extract_prompt(chosen, rejected):
    min_len = min(len(chosen), len(rejected))
    idx = 0
    while idx < min_len and chosen[idx] == rejected[idx]:
        idx += 1
    return chosen[:idx].strip()


def tokenize_pair(tokenizer, prompts, outputs, device, max_length=512):
    texts = [p + "\n" + o for p, o in zip(prompts, outputs)]
    tok = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return tok["input_ids"].to(device), tok["attention_mask"].to(device)


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

    optimizer = AdamW(policy.parameters(), lr=1e-5)

    # ---------------------------
    # Load dataset
    # ---------------------------
    raw = load_dataset("Anthropic/hh-rlhf")["train"].select(range(512))

    # Extract prompt, chosen, rejected
    prompts = []
    chosens = []
    rejects = []

    for x in raw:
        chosen = x["chosen"]
        rejected = x["rejected"]
        prompt = extract_prompt(chosen, rejected)

        chosen_ans = chosen[len(prompt):].strip()
        rejected_ans = rejected[len(prompt):].strip()

        prompts.append(prompt)
        chosens.append(chosen_ans)
        rejects.append(rejected_ans)

    batch_size = 4

    for step in range(200):
        idx = torch.randint(0, len(prompts), (batch_size,))

        batch_prompts = [prompts[i] for i in idx]
        batch_chosen = [chosens[i] for i in idx]
        batch_reject = [rejects[i] for i in idx]

        input_ch, attn_ch = tokenize_pair(tokenizer, batch_prompts, batch_chosen, device)
        input_re, attn_re = tokenize_pair(tokenizer, batch_prompts, batch_reject, device)

        logp_ch = compute_seq_logprobs(policy, input_ch, attn_ch)
        logp_re = compute_seq_logprobs(policy, input_re, attn_re)

        with torch.no_grad():
            logp_ref_ch = compute_seq_logprobs(ref_policy, input_ch, attn_ch)
            logp_ref_re = compute_seq_logprobs(ref_policy, input_re, attn_re)

        loss = dpo_loss(logp_ch, logp_re, logp_ref_ch, logp_ref_re, beta=0.1)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        if step % 10 == 0:
            print(f"[Step {step}] loss={loss.item():.4f}")

    torch.save(policy.state_dict(), "policy_dpo.pt")
    print("Saved policy_dpo.pt")


if __name__ == "__main__":
    main()
