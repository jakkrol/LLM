import json
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import GPT2Tokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, GPT2LMHeadModel 
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# # -----------------------------
# # Load tokenizer & base model
# # -----------------------------
# model_name = "models/gpt2_local"
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token  # GPT2 has no pad token

# model = GPT2LMHeadModel.from_pretrained(model_name)

# # -----------------------------
# # Apply LoRA
# # -----------------------------
# lora_config = LoraConfig(
#     r=8,                   # Low-rank matrices dimension
#     lora_alpha=16,          # Scaling factor
#     target_modules=["c_attn"],  # GPT2 attention projection layers
#     lora_dropout=0.1,
#     bias="none",
#     task_type=TaskType.CAUSAL_LM
# )

# model = get_peft_model(model, lora_config)

# # -----------------------------
# # Prepare dataset
# # -----------------------------
# class TwitchDataset(Dataset):
#     def __init__(self, path, tokenizer, block_size=128):
#         self.examples = []
#         with open(path, "r", encoding="utf-8") as f:
#             for line in f:
#                 data = json.loads(line)
#                 text = data["text"].strip()
#                 if text:
#                     tokenized = tokenizer(
#                         text,
#                         truncation=True,
#                         max_length=block_size,
#                         padding="max_length",   # <- pad to block_size
#                         return_tensors="pt"
#                     )
#                     self.examples.append(tokenized.input_ids.squeeze(0))

#     def __len__(self):
#         return len(self.examples)

#     def __getitem__(self, idx):
#         return {"input_ids": self.examples[idx], "labels": self.examples[idx]}

# train_dataset = TwitchDataset("data/twitch_data.jsonl", tokenizer)
# # Optionally split 80/20 train/test
# train_size = int(0.8 * len(train_dataset))
# test_size = len(train_dataset) - train_size
# train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])

# # -----------------------------
# # Training setup
# # -----------------------------
# training_args = TrainingArguments(
#     output_dir="./gpt2_lora",
#     num_train_epochs=3,
#     per_device_train_batch_size=4,
#     gradient_accumulation_steps=8,
#     learning_rate=3e-4,
#     fp16=True,
#     save_strategy="epoch",
#     logging_steps=50,
#     report_to="none"  # disable wandb/other logging
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=test_dataset
# )

# # -----------------------------
# # Train LoRA adapters
# # -----------------------------
# trainer.train()

# # -----------------------------
# # Save LoRA adapters
# # -----------------------------
# model.save_pretrained("./gpt2_lora")
# tokenizer.save_pretrained("./gpt2_lora")







# base_model = GPT2LMHeadModel.from_pretrained("models/gpt2_local")  # or the original GPT2 base

# # Load the previously trained LoRA adapter
# model = PeftModel.from_pretrained(base_model, "models/gpt2_convo6EpLora")

# # Load tokenizer
# tokenizer = GPT2Tokenizer.from_pretrained("models/gpt2_convo6EpLora")
tokenizer = GPT2Tokenizer.from_pretrained("models/gpt2_local")
model = GPT2LMHeadModel.from_pretrained("models/gpt2_local")


# -----------------------------
# Apply LoRA
# -----------------------------
lora_config = LoraConfig(
    r=8,                   # Low-rank matrices dimension
    lora_alpha=16,          # Scaling factor
    target_modules=["c_attn"],  # GPT2 attention projection layers
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)



texts = []
with open("data/twitch_data.jsonl", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        texts.append(data["text"])
        # q = data["question"]
        # a = data["answer"]
        # texts.append(f"User: {q}\nAI: {a}")

# print(f"Loaded {len(texts)} conversation pairs.")
# print("Sample:", texts[0])

full_text = "\n".join(texts)
twitch_chat_data = tokenizer.encode(full_text, add_special_tokens=False)


total_length = len(twitch_chat_data)
train_data = twitch_chat_data[:int(total_length * 0.8)]
test_data = twitch_chat_data[int(total_length * 0.8):]

train_data = torch.tensor(train_data, dtype=torch.long)
test_data = torch.tensor(test_data, dtype=torch.long)

print(train_data[:10])



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


block_size = 128
def create_blocks(data, block_size):
    num_blocks = len(data) // block_size
    data = data[:num_blocks * block_size]
    data = data.view(num_blocks, block_size)
    return data


train_data = create_blocks(train_data, block_size)
test_data = create_blocks(test_data, block_size)


batch_size = 4

train_dataset = TensorDataset(train_data, train_data)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

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

model.save_pretrained("models/gpt2_twitchLora")
tokenizer.save_pretrained("models/gpt2_twitchLora")






test_dataset = TensorDataset(test_data, test_data)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

model.eval()
total_loss = 0
with torch.no_grad():
    for batch in test_loader:
        inputs, labels = [x.to(device) for x in batch]
        outputs = model(input_ids=inputs, labels=labels)
        total_loss += outputs.loss.item()

avg_loss = total_loss / len(test_loader)
print(f"Test loss: {avg_loss}")
