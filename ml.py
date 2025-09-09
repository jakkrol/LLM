from transformers import GPT2Tokenizer, GPT2Model
import tiktoken
import csv
import torch

# tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir="./gpt2_local")
# model = GPT2Model.from_pretrained("gpt2", cache_dir="./gpt2_local")


# load data csv
twitch_chat_data = ""
with open("data/twitch_data.csv", newline='', encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if(row["message"]):
            twitch_chat_data += row["message"] + "\n"

enc = tiktoken.get_encoding("gpt2")

twitch_chat_data = enc.encode(twitch_chat_data)

total_length = len(twitch_chat_data)
train_data = twitch_chat_data[:int(total_length * 0.8)]
test_data = twitch_chat_data[int(total_length * 0.8):]

train_data = torch.tensor(train_data)
test_data = torch.tensor(test_data)

print(train_data[:10])