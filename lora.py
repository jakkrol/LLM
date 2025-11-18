from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
import tiktoken
import csv
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import os
import json
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from peft import LoraConfig, get_peft_model, TaskType, PeftModel


def collate_fn(batch):
    input_ids, labels = zip(*batch)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    return input_ids, labels

# --- Load base GPT-2 ---
tokenizer = GPT2Tokenizer.from_pretrained("models/gpt2_local")
model = GPT2LMHeadModel.from_pretrained("models/gpt2_local")
# model_A = PeftModel.from_pretrained(model, "models/lora_adapter_A")
# model_B = PeftModel.from_pretrained(model, "models/lora_adapter_B")


# -----------------------------
# Apply LoRA
# -----------------------------
lora_config = LoraConfig(
    r=8,                   # Low-rank matrices dimension
    lora_alpha=16,          # Scaling factor
    target_modules=["c_attn", "c_proj"],  # GPT2 attention projection layers
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)


tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# --- Prepare data ---
class ConversationDataset(Dataset):
    def __init__(self, path, tokenizer, block_size=128):
        self.samples = []

        with open(path, encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                q = data["question"].strip()
                a = data["answer"].strip()

                # INPUT: User question + AI cue
                prompt = f"User: {q}\nAI:"
                # LABELS: AI answer (we add EOS for clarity)
                target = f" {a}{tokenizer.eos_token}"

                enc_prompt = tokenizer.encode(prompt, add_special_tokens=False)
                enc_target = tokenizer.encode(target, add_special_tokens=False)

                input_ids = enc_prompt + enc_target
                labels = [-100] * len(enc_prompt) + enc_target  # ignore loss for prompt tokens

                if len(input_ids) > block_size:
                    input_ids = input_ids[:block_size]
                    labels = labels[:block_size]

                self.samples.append({
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long)
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]["input_ids"], self.samples[idx]["labels"]

# --- Create dataset and loader ---
dataset = ConversationDataset("data/Conversation.jsonl", tokenizer, block_size=128)
# train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
train_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# --- Optimizer ---
optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)

# --- Training ---
model.train()
epochs = 6

for epoch in range(epochs):
    total_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(input_ids=inputs, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss / len(train_loader):.4f}")

# --- Save fine-tuned model ---
model.save_pretrained("models/gpt2_HATER")
tokenizer.save_pretrained("models/gpt2_HATER")










