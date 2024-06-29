import math
import torch
import torch.nn as nn

from torch.nn import functional as F

# ---------------------------------------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """
        Causal self-attention layer. The layer is masked to prevent attending to the future.
        !!!NOTE: Attention is a communication mechanism between different parts of the input sequence.
    """
    def __init__(self, config):
        super().__init__()

        assert config.n_embd % config.n_head == 0

        # Key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.PARVUS_SCALE_INIT = 1

        # Regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # Not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(
            config.block_size, 
            config.block_size
        )).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # These are the batch size, sequence length, embedding dimensionality (n_embd)

        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Attention (materializes the large (T, T) matrix for all the queries and keys)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # Output projection
        y = self.c_proj(y)

        return y

# ---------------------------------------------------------------------------------------------------------

class MLP(nn.Module):
    """
        Simple MLP with GELU activation.
        !!!NOTE: In the MLP there is no communication between different parts of the input sequence.
    """
    def __init__(self, config):
        super().__init__()

        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu   = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.PARVUS_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)

        return x

# ---------------------------------------------------------------------------------------------------------

class Block(nn.Module):
    """Transformer block"""
    def __init__(self, config):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd)  # Layer normalization 
        self.attn = CausalSelfAttention(config)  # Causal self-attention layer
        self.ln_2 = nn.LayerNorm(config.n_embd)  # Layer normalization
        self.mlp  = MLP(config)                  # Feed-forward layer or Multi-Layer Perceptron

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
