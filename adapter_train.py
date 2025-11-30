import json
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from peft import LoraConfig, get_peft_model, TaskType
import torch.nn as nn

# -----------------------------
# Load tokenizer & base model
# -----------------------------
tokenizer = GPT2Tokenizer.from_pretrained("test_models/gpt2_convo6Ep")
tokenizer.pad_token = tokenizer.eos_token

base_model = GPT2LMHeadModel.from_pretrained("test_models/gpt2_convo6Ep")

# -----------------------------
# Apply REAL LoRA (correct)
# -----------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],  # GPT-2 attention projection layers
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(base_model, lora_config)

# IMPORTANT: Verify LoRA is the *only* trainable params
model.print_trainable_parameters()   # MUST show ~0.2% trainable

# -----------------------------
# Load & prep dataset
# -----------------------------
texts = []
with open("data/HateSpeech.jsonl", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        texts.append(data["text"])

full_text = "\n".join(texts)
encoded = tokenizer.encode(full_text, add_special_tokens=False)

total_length = len(encoded)
train_data = torch.tensor(encoded[:int(total_length * 0.8)], dtype=torch.long)
test_data = torch.tensor(encoded[int(total_length * 0.8):], dtype=torch.long)

block_size = 128
def create_blocks(data, block):
    n = len(data) // block
    data = data[:n * block]
    return data.view(n, block)

train_data = create_blocks(train_data, block_size)
test_data = create_blocks(test_data, block_size)

train_dataset = TensorDataset(train_data, train_data)
test_dataset = TensorDataset(test_data, test_data)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4)

# -----------------------------
# Training LoRA ONLY
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

epochs = 3
for epoch in range(epochs):
    for batch in train_loader:
        inputs, labels = [b.to(device) for b in batch]

        outputs = model(input_ids=inputs, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} | Loss: {loss.item()}")

# -----------------------------
# SAVE ONLY THE LORA ADAPTER
# -----------------------------
model.save_pretrained("models/gpt2_convoHlora")
tokenizer.save_pretrained("models/gpt2_convoHlora")


