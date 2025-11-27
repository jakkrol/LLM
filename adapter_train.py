import json
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from peft import LoraConfig, get_peft_model, TaskType

# -----------------------------
# Load tokenizer & base model
# -----------------------------
model_name = "models/gpt2_local"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # GPT2 nie ma pad_token
base_model = GPT2LMHeadModel.from_pretrained(model_name)

# -----------------------------
# Apply LoRA
# -----------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(base_model, lora_config)

# -----------------------------
# Load and tokenize data
# -----------------------------
texts = []
with open("data/twitch_data.jsonl", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        texts.append(data["text"])

full_text = "\n".join(texts)
encoded_data = tokenizer.encode(full_text, add_special_tokens=False)
encoded_data = torch.tensor(encoded_data, dtype=torch.long)

# Split train/test
total_length = len(encoded_data)
train_data = encoded_data[:int(total_length * 0.8)]
test_data = encoded_data[int(total_length * 0.8):]

# Create blocks
block_size = 128
def create_blocks(data, block_size):
    num_blocks = len(data) // block_size
    data = data[:num_blocks * block_size]
    return data.view(num_blocks, block_size)

train_data = create_blocks(train_data, block_size)
test_data = create_blocks(test_data, block_size)

# Dataloaders
batch_size = 4
train_loader = DataLoader(TensorDataset(train_data, train_data), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(test_data, test_data), batch_size=batch_size)

# -----------------------------
# Training
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
model.train()

epochs = 4
for epoch in range(epochs):
    for batch in train_loader:
        inputs, labels = [x.to(device) for x in batch]
        outputs = model(input_ids=inputs, labels=labels)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch: {epoch}, Loss: {loss.item()}")

# -----------------------------
# Save ONLY LoRA adapter
# -----------------------------
model.save_pretrained("models/gpt2_lora_adapter")  # <- tylko adapter
tokenizer.save_pretrained("models/gpt2_lora_adapter")  # tokenizer potrzebny do użycia adaptera

# -----------------------------
# Evaluation
# -----------------------------
model.eval()
total_loss = 0
with torch.no_grad():
    for batch in test_loader:
        inputs, labels = [x.to(device) for x in batch]
        outputs = model(input_ids=inputs, labels=labels)
        total_loss += outputs.loss.item()

avg_loss = total_loss / len(test_loader)
print(f"Test loss: {avg_loss}")
