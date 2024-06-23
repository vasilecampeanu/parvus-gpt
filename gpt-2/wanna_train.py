import torch
from awesome_gpt import GPT, GPTConfig
from data_loader_lite import DataLoaderLite
import time

device = 'mps'
model = GPT(GPTConfig())
model.to(device)
train_loader = DataLoaderLite(B=4, T=32)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for i in range(50):
    t0 = time.time()

    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)

    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()

    torch.mps.synchronize() # Wait for the GPU to finish work

    t1 = time.time()

    dt = (t1 - t0) * 1000 # Time difference in miliseconds
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)

    print(f"loss at step {i:02} | loss: {loss.item():18.15f} | dt: {dt:6.2f}ms | tok/sec: {tokens_per_sec:6.2f}")
