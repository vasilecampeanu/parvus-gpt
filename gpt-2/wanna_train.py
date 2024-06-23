from torch.nn import functional as F
import tiktoken
import torch
from gpt import GPT, GPTConfig

# Get a data batch

encoder = tiktoken.get_encoding('gpt2')

with open('../datasets/tinyshakespeare.txt', 'r') as f:
    text = f.read()

data = text[:1000]
tockens = encoder.encode(data)

B, T = 4, 32

buf = torch.tensor(tockens[:B*T + 1])

buf = buf.to('mps')

x = buf[:-1].view(B, T)
y = buf[1:].view(B, T)

# Model initialization

model = GPT(GPTConfig())
model.to('mps')
logits, loss = model(x, y)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for i in range(50):
    # We need to start with a zero gradient
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()

    # tensor -> cpu -> float
    print(f"Loss at iteration {i:02}: {loss.item()}")
