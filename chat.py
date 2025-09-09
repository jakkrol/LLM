import tiktoken

enc = tiktoken.get_encoding("gpt2")

encoding = enc.encode("I have a secret recipe for chicken nuggets that I'm not supposed to share becouse it's a family secret.")

print(encoding)