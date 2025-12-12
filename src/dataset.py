from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

class PreferenceDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=512):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        chosen = ex["chosen"]
        rejected = ex["rejected"]

        tc = self.tokenizer(
            chosen,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        tr = self.tokenizer(
            rejected,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "chosen_input_ids": tc["input_ids"].squeeze(0),
            "chosen_attention_mask": tc["attention_mask"].squeeze(0),
            "rejected_input_ids": tr["input_ids"].squeeze(0),
            "rejected_attention_mask": tr["attention_mask"].squeeze(0),
        }

def build_debug_loader(model_name="distilgpt2", batch_size=2):
    ds = load_dataset("Anthropic/hh-rlhf")["train"].select(range(20))  # 只取 20 条做 debug
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pref_ds = PreferenceDataset(ds, tokenizer, max_length=256)
    loader = DataLoader(pref_ds, batch_size=batch_size, shuffle=True)
    return loader

if __name__ == "__main__":
    loader = build_debug_loader()
    batch = next(iter(loader))
    for k, v in batch.items():
        print(k, v.shape)
