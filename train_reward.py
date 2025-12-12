# train_reward.py
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import random_split, DataLoader
from transformers import AutoTokenizer

from src.dataset import PreferenceDataset
from src.reward_model import RewardModel
from datasets import load_dataset

def pairwise_loss(r_chosen, r_rejected):
    return -F.logsigmoid(r_chosen - r_rejected).mean()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = "distilgpt2"

    raw = load_dataset("Anthropic/hh-rlhf")["train"].select(range(2000))  # 先 2000 条 debug
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = PreferenceDataset(raw, tokenizer, max_length=256)
    n = len(dataset)
    n_val = int(0.1 * n)
    n_train = n - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False)

    model = RewardModel(base_model_name=base_model).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-5)

    for epoch in range(2):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            r_c = model(batch["chosen_input_ids"], batch["chosen_attention_mask"])
            r_r = model(batch["rejected_input_ids"], batch["rejected_attention_mask"])

            loss = pairwise_loss(r_c, r_r)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * r_c.size(0)
            correct += (r_c > r_r).sum().item()
            total += r_c.size(0)

        print(f"[Epoch {epoch}] train loss={total_loss/total:.4f}, acc={correct/total:.4f}")

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                r_c = model(batch["chosen_input_ids"], batch["chosen_attention_mask"])
                r_r = model(batch["rejected_input_ids"], batch["rejected_attention_mask"])
                val_correct += (r_c > r_r).sum().item()
                val_total += r_c.size(0)

        print(f"[Epoch {epoch}] val acc={val_correct/val_total:.4f}")

    torch.save(model.state_dict(), "reward_model.pt")
    print("Saved reward model to reward_model.pt")

if __name__ == "__main__":
    main()
