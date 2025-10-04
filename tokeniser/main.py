import tiktoken


enc = tiktoken.encoding_for_model("gpt-4o")

text = "Hey there! I am an AI assistant."

tokens = enc.encode(text)

print("Tokens: ", tokens)

print("Decoded tokens: ", enc.decode(tokens))