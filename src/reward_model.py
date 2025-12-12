# src/reward_model.py
import torch
import torch.nn as nn
from transformers import AutoModel

class RewardModel(nn.Module):
    def __init__(self, base_model_name="distilgpt2"):
        super().__init__()
        self.base = AutoModel.from_pretrained(base_model_name)
        hidden = self.base.config.hidden_size
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # [B, T, H]
        last_token = last_hidden[:, -1, :]
        reward = self.value_head(last_token).squeeze(-1)  # [B]
        return reward
