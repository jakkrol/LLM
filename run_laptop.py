# twitch_gpt2_finetune.py
import csv
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.optim import AdamW


# ---------------------------
# 1️⃣ Load GPT-2 locally
# ---------------------------
save_dir = "models/gpt2_local"
tokenizer = GPT2Tokenizer.from_pretrained(save_dir)
model = GPT2LMHeadModel.from_pretrained(save_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ---------------------------
# 2️⃣ Load and prepare Twitch chat data
# ---------------------------
chat_text = ""
with open("data/light.csv", newline='', encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if row["message"]:
            # Format as short chat lines
            chat_text += f"user: {row['message']}\n"

# Encode text
encoded = tokenizer.encode(chat_text, add_special_tokens=False)
train_size = int(0.8 * len(encoded))

train_data = torch.tensor(encoded[:train_size], dtype=torch.long)
test_data  = torch.tensor(encoded[train_size:], dtype=torch.long)

# ---------------------------
# 3️⃣ Split into blocks
# ---------------------------
block_size = 128

def create_blocks(data, block_size):
    num_blocks = len(data) // block_size
    data = data[:num_blocks * block_size]
    return data.view(num_blocks, block_size)

train_data = create_blocks(train_data, block_size)
test_data = create_blocks(test_data, block_size)

# ---------------------------
# 4️⃣ Create DataLoader
# ---------------------------
batch_size = 4
train_dataset = TensorDataset(train_data, train_data)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# ---------------------------
# 5️⃣ Optimizer
# ---------------------------
optimizer = AdamW(model.parameters(), lr=5e-5)

# ---------------------------
# 6️⃣ Fine-tune model
# ---------------------------
epochs = 5  # Increase for better results on small datasets
model.train()

for epoch in range(epochs):
    total_loss = 0
    for batch in train_loader:
        inputs, labels = [x.to(device) for x in batch]

        outputs = model(input_ids=inputs, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {total_loss/len(train_loader):.4f}")

# Save fine-tuned model
model.save_pretrained("models/gpt2_finetuned")
tokenizer.save_pretrained("models/gpt2_finetuned")

# ---------------------------
# 7️⃣ Generation example
# ---------------------------
model.eval()
prompt = "user: EZ Clap\nuser: "
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

output = model.generate(
    input_ids,
    max_length=100,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.8,
    pad_token_id=tokenizer.eos_token_id
)

print("\n--- Generated Chat ---\n")
print(tokenizer.decode(output[0], skip_special_tokens=True))
