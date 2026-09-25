# %% [markdown]
# ### **Download "The Verdict" Dataset**

# %%
import urllib.request
url = ("https://raw.githubusercontent.com/rasbt/"
       "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
       "the-verdict.txt")
file_path = "the-verdict.txt"
urllib.request.urlretrieve(url, file_path)

# %% [markdown]
# ## **Tokenizing Text**

# %% [markdown]
# **Tokenizing refers to individual words, special charecters, including punctuation charecter or part of a word**

# %% [markdown]
# **Tokenization**
# --> Split text (on whitespaces, comma, special charecter or part of a word)

# %%
# Reading in a short story as text sample into Python

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print("Total number of character:", len(raw_text))
print(raw_text[:99])

# %%
# Split text on whitespace charecter

import re
text = "Hello, world. This, is a test."
result = re.split(r'(\s)', text)
print(result)

# %%
# Split text on whitespace, commas, periods

result = re.split(r'([,.]|\s)', text)
print(result)

# %%
result = [item for item in result if item.strip()]
print(result)

# %%
# Split text on additional special charecter

text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
print(result)

# %%
# Apply on entire short story

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))

# %%
print(preprocessed[:30])

# %% [markdown]
# ## **Converting tokens into token IDs**

# %%
# Number of unique tokens

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)

# %%
# Creating a vocabulary

# all_words -> enumerate() -> (index, word) -> (word → index) -> vocab dictionary

vocab = {token:integer for integer,token in enumerate(all_words)}   # enumerate add a counter
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break   

# %%
# Create a simple tokenizer

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab           
        self.int_to_str = {i:s for s,i in vocab.items()}       
    def encode(self, text):        
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    def decode(self, ids):        
        text = " ".join([self.int_to_str[i] for i in ids])  #Converts token IDs back into text

        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)   # Removes spaces before the specified punctuation
        return text

# %%
tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know," 
       Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)

# %%
print(tokenizer.decode(ids))

# %%
# text = "Hello, do you like tea?"    # This should return a keyerror "Hello", because in training data it doesn't see this one.
# print(tokenizer.encode(text))

# %% [markdown]
# #### **Adding special context token**

# %%
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_tokens)}
print(len(vocab.items()))

# %%
for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)

# %%
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [item if item in self.str_to_int           
                        else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)   
        return text

# %%
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)

# %%
tokenizer = SimpleTokenizerV2(vocab)
print(tokenizer.encode(text))

# %%
print(tokenizer.decode(tokenizer.encode(text)))

# %% [markdown]
# #### **Byte Pair Encoding**

# %%
#pip install tiktoken

# %%
from importlib.metadata import version
import tiktoken
print("tiktoken version:", version("tiktoken"))

# %%
tokenizer = tiktoken.get_encoding("gpt2")

# %%
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

# %%
strings = tokenizer.decode(integers)
print(strings)

# %% [markdown]
# ##### **Exercise-2.1**

# %%
text = (
    "Akwirw ier"
)
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

# %%
strings = tokenizer.decode(integers)
print(strings)

# %% [markdown]
# ## **Data sampling with a sliding window**

# %%
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

# %%
enc_sample = enc_text[50:]

# %%
context_size = 4    #The context size determines how many tokens are included in the input.
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f"x: {x}")
print(f"y:      {y}")

# %%
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "---->", desired)

# %%
for i in range(1, context_size+1):
    context = enc_sample[:i]    # This is a list of number
    desired = enc_sample[i]     # This is a single integer
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))  # decode wants a list of number

# %%
# A dataset for batched inputs and targets

import torch
from torch.utils.data import Dataset, DataLoader
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)   #Tokenizes the entire text 38
        
        for i in range(0, len(token_ids) - max_length, stride):    #Uses a sliding window to chunk the book into overlapping sequences of max_length
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):   #Returns the total number of rows in the dataset
        return len(self.input_ids)
    
    def __getitem__(self, idx):        # Returns a single row from the dataset
        return self.input_ids[idx], self.target_ids[idx]

# %%
# A data loader to generate batches with input-with pairs

def create_dataloader_v1(txt, batch_size=4, max_length=256,
                 stride=128, shuffle=True, drop_last=True,
                 num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")             # Initializes the tokenizer           
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)  # Creates dataset
    dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    drop_last=drop_last,    # drop_last=True drops the last batch if it is shorter than the specified batch_size to prevent loss spikes during training.
    num_workers=num_workers    # The number of CPU processes to use for preprocessing
        )
    return dataloader

# %%
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
data_iter = iter(dataloader)     # Converts dataloader into a Python iterator to fetch the next entry via Python’s built-in next() function
first_batch = next(data_iter)
print(first_batch)

# %%
second_batch = next(data_iter)
print(second_batch)

# %%
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=4, stride=4, # here, max length = stride, so that we can prevent the overlap between batches
    shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)

# %% [markdown]
# ### **Creating Token Embeddings**

# %%
input_ids = torch.tensor([2, 3, 5, 1])

# %%
vocab_size = 6
output_dim = 3

# %%
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)

# %%
print(embedding_layer(torch.tensor([3])))

# %%
print(embedding_layer(input_ids))

# %% [markdown]
# ### **Encoding Word Positions**

# %%
vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# %%
max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
   stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

# %%
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

# %%
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)

# %%
input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)


