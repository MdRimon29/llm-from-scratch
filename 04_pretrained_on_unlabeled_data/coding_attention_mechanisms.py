# %% [markdown]
# ## **Different parts of the input with self-attention**

# %% [markdown]
# ### **A simple self-attention mechanism without trainable weights**

# %%
import torch
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

# %%
# intermediate attention scores between the query token and each input token
# computing the dot product of the query, x(2), with every other input token

query = inputs[1]        # The second input token serves as the query.                   
attn_scores_2 = torch.empty(inputs.shape[0])    # attn_scores_2 = [?, ?, ?, ?, ?, ?]

for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)

print(attn_scores_2)

# %%
#  Normalization is to obtain attention weights that sum up to 1

attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())

# %%
# Softmax function for normalization to managing extreme values and offers more favorable gradient properties during training

def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)
print("Attention weights:", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())

# %%
# pytorch softmax function

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())

# %%
#  context vector z(2)

query = inputs[1]        
context_vec_2 = torch.zeros(query.shape)

for i,x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i

print(context_vec_2)

# %% [markdown]
# ### **Computing attention weights for all input tokens**

# %%
attn_scores = torch.empty(6, 6)

for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)

print(attn_scores)

# %%
# matrix multiplication instead of for loops, for faster computation

attn_scores = inputs @ inputs.T
print(attn_scores)

# %%
# Normalize the values

attn_weights = torch.softmax(attn_scores, dim=-1)   #  dim=-1, we are instructing the softmax function to apply the normalization along the last dimension of the attn_scores tensor.
print(attn_weights)

# %%
# Verify normalization

row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("Row 2 sum:", row_2_sum)
print("All row sums:", attn_weights.sum(dim=-1))

# %%
# compute all context vector

all_context_vecs = attn_weights @ inputs
print(all_context_vecs)

# %% [markdown]
# ## **Implementing self-attention with trainable weights**

# %% [markdown]
# ### **Computing the attention weights step by step**

# %%
x_2 = inputs[1]    # The second input element
d_in = inputs.shape[1]    # The input embedding size, d=3 
d_out = 2  # The output embedding size, d_out=2

# %%
# Initialize three weight matrix

torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

# %%
# compute the query, key, and value vectors

query_2 = x_2 @ W_query 
key_2 = x_2 @ W_key 
value_2 = x_2 @ W_value
print(query_2)

# %%
# Obtain all keys and values

keys = inputs @ W_key 
values = inputs @ W_value

print("keys.shape:", keys.shape)
print("values.shape:", values.shape)

# %%
# attention score ω22

keys_2 = keys[1]            
attn_score_22 = query_2.dot(keys_2)
print(attn_score_22)

# %%
# generalize attention score

attn_scores_2 = query_2 @ keys.T      
print(attn_scores_2)

# %%
# Compute attention weight by scale the attention score

d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print(attn_weights_2)

# %%
# Compute the context vector

context_vec_2 = attn_weights_2 @ values
print(context_vec_2)

# %% [markdown]
# #### **A compact self-attention class**

# %%
import torch.nn as nn
class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        
        context_vec = attn_weights @ values
        return context_vec

# %%
torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))

# %%
#  A self-attention class using PyTorch’s Linear layers

class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)    # nn.Linear() has an optimized weight initialization scheme, contributing to more stable and effective model training
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    
    def forward(self, x):
        keys = self.W_key(x)    # internally calulates,  x @ W.T + bias
        queries = self.W_query(x)
        values = self.W_value(x)
        
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        
        context_vec = attn_weights @ values
        return context_vec

# %%
torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs))

# %% [markdown]
# #### **Applying a causal attention mask**

# %%
# compute the attention weights using the softmax function

queries = sa_v2.W_query(inputs)    
keys = sa_v2.W_key(inputs) 
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
print(attn_weights)

# %%
# create mask above diagonal using trill function

context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))    # torch.tril() means take the lower triangular part, including the diagonal
print(mask_simple)

# %%
#  multiply this mask with the attention weights to zero-out the values above the diagonal

masked_simple = attn_weights*mask_simple
print(masked_simple)

# %%
# renormalize the attention weights to sum up to 1

row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm)

# %%
# mask with 1s above the diagonal and then replacing these 1s with negative infinity (-inf) values

mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)   # torch.triu() means take the upper triangular part, excluding the diagonal
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)

# %%
attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
print(attn_weights)

# %% [markdown]
# #### **Masking additional attention weights with dropout**

# %%
# PyTorch dropout implementation

torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)   
example = torch.ones(6, 6)     
print(dropout(example))

# %%
# apply dropout to the attention weight matrix itself

torch.manual_seed(123)
print(dropout(attn_weights))

# %% [markdown]
# #### **Implementing a compact causal attention class**

# %%
batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape)  

# %%
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)           
        self.register_buffer(      # buffers are automatically moved to the appropriate device (CPU or GPU) along with our model
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape                  
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)   
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf) # operations with a trailing underscore(_) are performed in-place, avoiding unnecessary memory copies.
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)
        
        context_vec = attn_weights @ values
        return context_vec

# %%
torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
print("context_vecs.shape:", context_vecs.shape)

# %% [markdown]
# ### **Extending single-head attention to multi-head attention**

# %%
# Stacking multiple single-head attention layers

class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(     # nn.ModuleList tells PyTorch that these are actual submodules whose parameters need to be tracked.
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) 
             for _ in range(num_heads)]
        )
    
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

# %%
# we can use the MultiHeadAttention Wrapper class

torch.manual_seed(123)
context_length = batch.shape[1] # This is the number of tokens
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)

# %% [markdown]
# #### **An efficient multi-head attention class**

# %%
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads   # Reduces the projection dim to match the desired output dim
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)   # Uses a Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        # Tensor shape: (b, num_tokens, d_out)
        keys = self.W_key(x)        
        queries = self.W_query(x)   
        values = self.W_value(x) 
        
        # We implicitly split the matrix by adding a num_heads dimension. Then we unroll the last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim).
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)      
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)  
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)                                                                   
        
        # Transposes from shape (b, num_tokens, num_heads, head_dim) to (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)         
        queries = queries.transpose(1, 2)   
        values = values.transpose(1, 2) 

        attn_scores = queries @ keys.transpose(2, 3)  # Computes dot product for each head
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]   # Masks truncated to the number of tokens
        
        attn_scores.masked_fill_(mask_bool, -torch.inf)    
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context_vec = (attn_weights @ values).transpose(1, 2)  # Tensor shape: (b, num_tokens, n_heads, head_dim)
        
        context_vec = context_vec.contiguous().view(    # Combines heads, where self.d_out = self.num_heads * self.head_dim
            b, num_tokens, self.d_out
        )
        context_vec = self.out_proj(context_vec)   # Adds an optional linear projection
        return context_vec

# %%
# Batch matrix multiplication

a = torch.tensor([[[[0.2745, 0.6584, 0.2775, 0.8573],   
                    [0.8993, 0.0390, 0.9268, 0.7388],
                    [0.7179, 0.7058, 0.9156, 0.4340]],
                   [[0.0772, 0.3565, 0.1479, 0.5331],
                    [0.4066, 0.2318, 0.4545, 0.9737],
                    [0.4606, 0.5159, 0.4220, 0.5786]]]])

# %%
print(a @ a.transpose(2, 3))

# %%
# Matrix multiplication for each head separately

first_head = a[0, 0, :, :]
first_res = first_head @ first_head.T
print("First head:\n", first_res)

second_head = a[0, 1, :, :]
second_res = second_head @ second_head.T
print("\nSecond head:\n", second_res)

# %%
# Use MultiHeadAttention class

torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)


