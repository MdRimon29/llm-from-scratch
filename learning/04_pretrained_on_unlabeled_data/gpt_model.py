# %% [markdown]
# ## **Coding an LLM architecture**

# %% [markdown]
# #### **Specify the configuration**

# %%
GPT_CONFIG_124M = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 768,          # Embedding dimension
    "n_heads": 12,           # Number of attention heads
    "n_layers": 12,          # Number of layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False        # Query-Key-Value bias
}

# %% [markdown]
# #### **A placeholder GPT model architecture class**

# %%
import torch
import torch.nn as nn
class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])])    # Uses a placeholder for TransformerBlock                                         
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])    # Uses a placeholder for LayerNorm
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
        

class DummyTransformerBlock(nn.Module):   
    def __init__(self, cfg):
        super().__init__()
    
    def forward(self, x):    
        return x
    

class DummyLayerNorm(nn.Module):          
    def __init__(self, normalized_shape, eps=1e-5):   
        super().__init__()
    
    def forward(self, x):
        return x

# %%
# tokenize a batch consisting of two text inputs

import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
batch = []

txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch)


# %%
# Initialize our dummy gpt model

torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
print("Output shape:", logits.shape)
print(logits)

# %% [markdown]
# ## **Normalizing activations with layer normalization**

# %%
# Implement a neural network layer with 5 inputs and 6 outputs

torch.manual_seed(123)
batch_example = torch.randn(2, 5)    
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
print("Batch Example: ", batch_example)
print("Output: ", out)

# %%
# Calculate mean and variance 

mean = out.mean(dim=-1, keepdim=True)   # Here, dim=-1 calculates mean across the column dimension to obtain one mean per row
var = out.var(dim=-1, keepdim=True)
print("Mean:\n", mean)
print("Variance:\n", var)

# %%
# Apply layer normalization

out_norm = (out - mean) / torch.sqrt(var)
mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)
print("Normalized layer outputs:\n", out_norm)
print("Mean:\n", mean)
print("Variance:\n", var)

# %%
torch.set_printoptions(sci_mode=False) # turn off the scientific notation when printing, for better readability
print("Mean:\n", mean)
print("Variance:\n", var)

# %% [markdown]
# #### **A layer normalization class**

# %%
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

# %%
# try LayerNorm module in practise

ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
print("Mean:\n", mean)
print("Variance:\n", var)

# %% [markdown]
# ## **Feed forward network with GELU activations**

# %% [markdown]
# #### **An implementation of the GELU activation function**

# %%
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))

# %%
# Compare GELU and ReLu

import matplotlib.pyplot as plt
gelu, relu = GELU(), nn.ReLU()

x = torch.linspace(-3, 3, 100)    # Creates 100 sample data points in the range –3 to 3
y_gelu, y_relu = gelu(x), relu(x)
plt.figure(figsize=(8, 3))
for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
    plt.subplot(1, 2, i)
    plt.plot(x, y)
    plt.title(f"{label} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# #### **A feed forward neural network module**

# %%
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )
    
    def forward(self, x):
        return self.layers(x)

# %%
# initialize FeedForward module

ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2, 3, 768)         
out = ffn(x)
print(out.shape)

# %% [markdown]
# ## **Shortcut connections**

# %% [markdown]
# Shortcut connections add the original input directly to the layer output (x + F(x)), creating a direct gradient path that helps prevent gradients from becoming extremely small in deep networks.

# %%
# A neural network to illustrate shortcut connections

class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
        ])

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)        # Compute the output of the current layer
            if self.use_shortcut and x.shape == layer_output.shape:   # Check if shortcut can be applied
                x = x + layer_output
            else:
                x = layer_output
        return x

# %%
# Initialize a layer without shortcut connection

layer_sizes = [3, 3, 3, 3, 3, 1]  
sample_input = torch.tensor([[1., 0., -1.]])
torch.manual_seed(123)                           
model_without_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=False
)

# %%
# implement a function that computes the gradients in the model’s backward pass:

def print_gradients(model, x):
    output = model(x)            
    target = torch.tensor([[0.]])
    
    loss = nn.MSELoss()
    loss = loss(output, target)   
    
    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")

# %%
print_gradients(model_without_shortcut, sample_input)

# %%
# instantiate a model with skip connections

torch.manual_seed(123)
model_with_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=True
)
print_gradients(model_with_shortcut, sample_input)

# %% [markdown]
# ## **Transformer block**

# %%
from coding_attention_mechanisms import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], 
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
    
    def forward(self, x):
        shortcut = x        # Shortcut connection for attention block
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut     # Add the original input back
        
        shortcut = x        # Shortcut connection for feed forward block
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut     # Add the original input back
        return x

# %%
# initialize transformer block

torch.manual_seed(123)
x = torch.rand(2, 4, 768)                  
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)
print("Input shape:", x.shape)
print("Output shape:", output.shape)

# %% [markdown]
# ## **The GPT model architecture implementation**

# %%
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        pos_embeds = self.pos_emb( torch.arange(seq_len, device=in_idx.device))  # The device setting will allow us to train the model on a CPU or GPU, depending on which device the input data sits on. 
        
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

# %%
# initialize the 124-million-parameter GPT model

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)

out = model(batch)
print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)
print(out)

# %%
# calculate total number of parameters

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")

# %%
print("Token embedding layer shape:", model.tok_emb.weight.shape)
print("Output layer shape:", model.out_head.weight.shape)

# %%
total_params_gpt2 = (
    total_params - sum(p.numel()
    for p in model.out_head.parameters())
)
print(f"Number of trainable parameters "
      f"considering weight tying: {total_params_gpt2:,}"
)

# %%
# Computes memory requirements of 163M parameters model

total_size_bytes = total_params * 4      # Calculates the total size in bytes (assuming float32, 4 bytes per parameter)
total_size_mb = total_size_bytes / (1024 * 1024)    # Converts to megabytes
print(f"Total size of the model: {total_size_mb:.2f} MB")

# %% [markdown]
# ## **Generating text**

# %%
# A function for the GPT model to generate text

def generate_text_simple(model, idx, max_new_tokens, context_size):  # idx is a (batch, n_tokens) array of indices in the current context.
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]   # Crops current context if it exceeds the supported context size, e.g., if LLM supports only 5 tokens, and the context size is 10, then only the last 5 tokens are used as context
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]   # Focuses only on the last time step, so that (batch, n_token, vocab_size) becomes (batch, vocab_size)
        probas = torch.softmax(logits, dim=-1)          # probas has shape (batch, vocab_size).
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)   # idx_next has shape (batch, 1).
        idx = torch.cat((idx, idx_next), dim=1)    # Appends sampled index to the running sequence, where idx has shape (batch, n_tokens+1)
    return idx

# %%
# we encode the input context into token IDs

start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print("encoded:", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)   # Adds batch dimension
print("encoded_tensor.shape:", encoded_tensor.shape)

# %%
# Output token

model.eval()                 
out = generate_text_simple(
    model=model,
    idx=encoded_tensor, 
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)
print("Output:", out)
print("Output length:", len(out[0]))

# %%
decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)


