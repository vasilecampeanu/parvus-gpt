import math
import torch
import torch.nn as nn

from torch.nn import functional as F

# ---------------------------------------------------------------------------------------------------------

class SelfAttentionMechanism(nn.Module):
    """
        Causal self-attention layer. The layer is masked to prevent attending to the future.
        !!!NOTE: Attention is a communication mechanism between different parts of the input sequence.
    """
    def __init__(self, config):
        super().__init__()

        assert config.n_embds % config.n_heads == 0

        # Key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embds, 3 * config.n_embds)

        # Output projection
        self.c_proj = nn.Linear(config.n_embds, config.n_embds)
        self.c_proj.PARVUS_SCALE_INIT = 1

        # Regularization
        self.n_heads = config.n_heads
        self.n_embds = config.n_embds

        # Not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(
            config.context_length, 
            config.context_length
        )).view(1, 1, config.context_length, config.context_length))

    def forward(self, x):
        B, T, C = x.size() # These are the batch size, sequence length, embedding dimensionality (n_embds)

        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_heads=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embds, dim=2)

        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)

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

class MultiLayerPerceptron(nn.Module):
    """
        Simple MLP with GELU activation.
        !!!NOTE: In the MLP there is no communication between different parts of the input sequence.
    """
    def __init__(self, config):
        super().__init__()

        self.c_fc   = nn.Linear(config.n_embds, 4 * config.n_embds)
        self.gelu   = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embds, config.n_embds)
        self.c_proj.PARVUS_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)

        return x

# ---------------------------------------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embds)   # Layer normalization
        self.attn = SelfAttentionMechanism(config) # Self-attention mechanism
        self.ln_2 = nn.LayerNorm(config.n_embds)   # Layer normalization
        self.mlp  = MultiLayerPerceptron(config)   # Multi-layer perceptron

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp (self.ln_2(x))
        return x
