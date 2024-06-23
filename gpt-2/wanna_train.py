import torch
from awesome_gpt import GPT, GPTConfig
from data_loader_lite import DataLoaderLite
import time

device = "mps"

torch.manual_seed(1337)

if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    torch.mps.manual_seed(1337)

train_loader = DataLoaderLite(B=4, T=1024)
torch.set_float32_matmul_precision('high')

model = GPT(GPTConfig())
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for i in range(50):
    t0 = time.time()

    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)

    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()

    # Wait for the GPU to finish work

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()

    t1 = time.time()

    # Time difference in miliseconds
    dt = (t1 - t0) * 1000
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)

    print(f"loss at step {i:02} | loss: {loss.item():18.15f} | dt: {dt:8.2f}ms | tok/sec: {tokens_per_sec:6.2f}")
