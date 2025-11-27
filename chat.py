import tiktoken

enc = tiktoken.get_encoding("gpt2")

encoding = enc.encode("dog")
#encoding = enc.encode("Electroencephalogram")

print(encoding)
for t in encoding:
    print(enc.decode([t]))