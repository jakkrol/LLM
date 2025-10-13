from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
import tiktoken
import csv
import torch
from torch.utils.data import DataLoader, TensorDataset
import os


# save_dir = "models/gpt2_local"

# # Create the folder if it doesn't exist
# os.makedirs(save_dir, exist_ok=True)

# # 1️⃣ Download GPT-2 tokenizer and model directly
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# model = GPT2LMHeadModel.from_pretrained("gpt2")

# # 2️⃣ Save them to your clean folder
# tokenizer.save_pretrained(save_dir)
# model.save_pretrained(save_dir)

# print(f"GPT-2 saved locally in {save_dir}")


tokenizer = GPT2Tokenizer.from_pretrained("models/gpt2_local")
model = GPT2LMHeadModel.from_pretrained("models/gpt2_local")



# load data csv
twitch_chat_data = ""
with open("data/light.csv", newline='', encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if(row["message"]):
            twitch_chat_data += row["message"] + "\n"


# enc = tiktoken.get_encoding("gpt2")
# twitch_chat_data = enc.encode(twitch_chat_data)
twitch_chat_data = tokenizer.encode(twitch_chat_data)


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

epochs = 1
for epoch in range(epochs):
    for batch in train_loader:
        inputs, labels = [x.to(device) for x in batch]

        outputs = model(input_ids=inputs, labels=labels)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch: {epoch}, Loss: {loss.item()}")

model.save_pretrained("models/gpt2_finetuned")
tokenizer.save_pretrained("models/gpt2_finetuned")









# test_dataset = TensorDataset(test_data, test_data)
# test_loader = DataLoader(test_dataset, batch_size=batch_size)

# model.eval()
# total_loss = 0
# with torch.no_grad():
#     for batch in test_loader:
#         inputs, labels = [x.to(device) for x in batch]
#         outputs = model(input_ids=inputs, labels=labels)
#         total_loss += outputs.loss.item()

# avg_loss = total_loss / len(test_loader)
# print(f"Test loss: {avg_loss}")