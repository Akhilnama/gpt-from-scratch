import requests
from typing import Dict, List
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text = requests.get(url).text
# print(f"lenght of dataset: {len(text)}")
# print(text[:500])

# vocab
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocab size:{vocab_size}")
# print(chars)

# mapping
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for ch,i in stoi.items()}

# print(stoi['a'])
# print(itos[stoi['a']])

def encode(s:str) -> List[int]:
    if not isinstance(s,str):
        raise TypeError(f"Expected 's' to be str, got {type(s)}")
    
    return [stoi[c] for c in s]

def decode(l):
    if not isinstance(l,list):
        raise TypeError(f"Expected 'l' to be list, got {type(l)}")
    
    return ''.join([itos[i] for i in l ])

sample = 'Akhil'
encoded = encode(sample)
decoded = decode(encoded)

# print(encoded,decoded)

# convert to tensor
import torch

data = torch.tensor(encode(text),dtype = torch.long)
# print(data.shape)
# print(data[:50])

# training pairs
block_size = 8
def get_batch(data):
    ix = torch.randint(len(data)-block_size,(1,))
    x = data[ix:ix+block_size]
    y = data[ix+1:ix+block_size+1]
    return x,y

x,y = get_batch(data)
# print(f"Input:{decode(x.tolist())}")
# print(f"Target:{decode(y.tolist())}")








# import requests
# import torch

# # Download dataset
# url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
# text = requests.get(url).text

# print("Dataset length:", len(text))

# # Build vocabulary
# chars = sorted(list(set(text)))
# vocab_size = len(chars)

# print("Vocabulary size:", vocab_size)

# stoi = {ch: i for i, ch in enumerate(chars)}
# itos = {i: ch for ch, i in stoi.items()}


# def encode(s):
#     return [stoi[c] for c in s]


# def decode(l):
#     return ''.join([itos[i] for i in l])


# # Convert full dataset to tensor
# data = torch.tensor(encode(text), dtype=torch.long)

# print("First 100 characters encoded:")
# print(data[:100])

# # Create sample batch
# block_size = 8
# ix = torch.randint(len(data) - block_size, (1,))
# x = data[ix:ix+block_size]
# y = data[ix+1:ix+block_size+1]

# print("Input sequence:", decode(x.tolist()))
# print("Target sequence:", decode(y.tolist()))