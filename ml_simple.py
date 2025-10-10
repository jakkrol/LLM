from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
import tiktoken
import csv
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
import json


tokenizer = GPT2Tokenizer.from_pretrained("models/gpt2_local")
model = GPT2LMHeadModel.from_pretrained("models/gpt2_local")



texts = []
with open("data/Conversation.jsonl", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        q = data["question"]
        a = data["answer"]
        texts.append(f"User: {q}\nAI: {a}")



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

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)




model.train()

epochs = 2
for epoch in range(epochs):
    for batch in train_loader:
        inputs, labels = [x.to(device) for x in batch]

        outputs = model(input_ids=inputs, labels=labels)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch: {epoch}, Loss: {loss.item()}")

model.save_pretrained("models/gpt2_convo")
tokenizer.save_pretrained("models/gpt2_convo")




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