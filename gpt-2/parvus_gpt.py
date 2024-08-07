import torch
import torch.nn as nn

from dataclasses import dataclass
from torch.nn import functional as F

# Local modules
from base import Block

@dataclass
class GPTConfig:
    context_length:  int = 1024  # max sequence length
    vocabulary_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layers: int        = 12    # number of layers
    n_heads:  int        = 12    # number of heads
    n_embds:  int        = 768   # embedding dimension

class ParvusGPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocabulary_size, config.n_embds),
            wpe  = nn.Embedding(config.context_length,  config.n_embds),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = nn.LayerNorm(config.n_embds),
        ))

        self.lm_head = nn.Linear(config.n_embds, config.vocabulary_size, bias=False)

        # Weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight
        
        # Init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'PARVUS_SCALE_INIT'):
                std *= (2 * self.config.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.context_length, f"Cannot forward sequence of length {T}, block size is only {self.config.context_length}"
        
        # Dorward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # Shape (T)
        pos_emb = self.transformer.wpe(pos) # Position embeddings of shape (T, n_embds)
        tok_emb = self.transformer.wte(idx) # Token embeddings of shape (B, T, n_embds)
        x = tok_emb + pos_emb
        
        # Forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        
        # Forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocabulary_size)

        # Loss computation
        loss = None

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("Loading weights from pretrained gpt: %s" % model_type)

        # n_layers, n_heads and n_embds are determined from model_type
        config_args = {
            'gpt2':         dict(n_layers=12, n_heads=12, n_embds=768),  # 124M  params
            'gpt2-medium':  dict(n_layers=24, n_heads=16, n_embds=1024), # 350M  params
            'gpt2-large':   dict(n_layers=36, n_heads=20, n_embds=1280), # 774M  params
            'gpt2-xl':      dict(n_layers=48, n_heads=25, n_embds=1600), # 1558M params
        }[model_type]

        config_args['vocabulary_size'] = 50257 # Always 50257 for GPT model checkpoints
        config_args['context_length'] = 1024  # Always  1024 for GPT model checkpoints
        
        # Create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = ParvusGPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # Discard this mask / buffer, not a param

        # Init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)

        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        # Basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # This means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # Special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # Vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model