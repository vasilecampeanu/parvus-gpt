import torch
from torch.nn import functional as F
import tiktoken
from base import GPT

num_return_sequences = 5
max_length = 30

model = GPT.from_pretrained('gpt2')
model.eval()
model.to('mps')

# Prefix tokens
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
x = tokens.to('mps')

# ---------------------------------------------------------------------------------------------------------

# Generate! Right now x is (B, T) where B = 5, T = 8
# Set the seed to 42
torch.manual_seed(42)
torch.mps.manual_seed(42)

while x.size(1) < max_length:
    # Forward the model to get the logits
    with torch.no_grad():
        logits = model(x) # (B, T, vocab_size)
        # Take the logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size)
        # Get the probabilities
        probs = F.softmax(logits, dim=-1)
        # Do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # Select a token from the top-k probabilities
        # NOTE: Multinomial does not demand the input to sum to 1
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        # Gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # Append to the sequence
        x = torch.cat((x, xcol), dim=1)

# Print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)