from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
import tiktoken
import csv
import torch
from torch.utils.data import DataLoader, TensorDataset
import os


# Load your fine-tuned model
# tokenizer = GPT2Tokenizer.from_pretrained("models/gpt2_local")
# model = GPT2LMHeadModel.from_pretrained("models/gpt2_local")

tokenizer = GPT2Tokenizer.from_pretrained("models/gpt2_test3")
model = GPT2LMHeadModel.from_pretrained("models/gpt2_test3")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # set to evaluation mode

# Example prompt
prompt = "User: What are you doing?\nAssistant:"

# Encode the prompt
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
attention_mask = torch.ones_like(input_ids)

# Generate text
output_ids = model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_length=50,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    pad_token_id=tokenizer.eos_token_id
)

# Decode generated text
stop_sequence = "User:"
generated_text = tokenizer.decode(output_ids[0])
generated_text = generated_text.split(stop_sequence)[1]  # cut at the next user

# generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(generated_text)



