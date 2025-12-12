# gen_winrate_samples.py
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def generate(model, tokenizer, prompts, device, max_new_tokens=128):
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

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = "distilgpt2"

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    # Load models
    ref_model = AutoModelForCausalLM.from_pretrained(base_model).to(device)

    def load_policy(weight_path):
        model = AutoModelForCausalLM.from_pretrained(base_model).to(device)
        state = torch.load(weight_path, map_location=device)
        model.load_state_dict(state)
        return model

    ppo_model  = load_policy("policy_ppo.pt")
    grpo_model = load_policy("policy_grpo.pt")
    dpo_model  = load_policy("policy_dpo.pt")

    # Load prompts from JSON
    with open("data/eval_prompts.json", "r", encoding="utf-8") as f:
        items = json.load(f)
    prompts = [x["prompt"] for x in items]

    # Generate outputs
    ref_outputs  = generate(ref_model,  tokenizer, prompts, device)
    ppo_outputs  = generate(ppo_model,  tokenizer, prompts, device)
    grpo_outputs = generate(grpo_model, tokenizer, prompts, device)
    dpo_outputs  = generate(dpo_model,  tokenizer, prompts, device)

    # Save results
    cases = []
    for i, p in enumerate(prompts):
        cases.append({
            "id": i,
            "prompt": p,
            "ref": ref_outputs[i],
            "ppo": ppo_outputs[i],
            "grpo": grpo_outputs[i],
            "dpo": dpo_outputs[i],
        })

    out_path = Path("winrate_cases.json")
    out_path.write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved {len(cases)} win-rate cases to {out_path}")


if __name__ == "__main__":
    main()
