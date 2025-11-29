from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
import tiktoken
import csv
import torch
from torch.utils.data import DataLoader, TensorDataset
from peft import PeftModel
import os


# Load your fine-tuned model
# tokenizer = GPT2Tokenizer.from_pretrained("models/gpt2_local")
# model = GPT2LMHeadModel.from_pretrained("models/gpt2_local")

tokenizer = GPT2Tokenizer.from_pretrained("models/gpt2_local")
model_base = GPT2LMHeadModel.from_pretrained("models/gpt2_local")
model = PeftModel.from_pretrained(model_base, "models/gpt2_lora_adapter")
print(model.print_trainable_parameters())
#model.merge_and_unload()
print(model.print_trainable_parameters())

# model_merge2 = PeftModel.from_pretrained(model_merge, "models/lora_adapter_B")
# model_merge2.merge_and_unload()


# model = model_merge2;
# model = model_merge;
# model = model_base;

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # set to evaluation mode

# Example prompt
prompt = 'User: Hello\nAI:'

# Encode the prompt
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
attention_mask = torch.ones_like(input_ids)

# Generate text
output_ids = model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_length=300,
    do_sample=True,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    pad_token_id=tokenizer.eos_token_id,
    repetition_penalty=1.2
)

# stop_sequence = "User:"
# generated_text = tokenizer.decode(output_ids[0])
# generated_text = generated_text.split(stop_sequence)[1]  # cut at the next user

generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(generated_text)



