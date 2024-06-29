import torch
from awesome_gpt import GPT, GPTConfig
from data_loader import DataLoader
import time
import csv

# mps - is refering to Metal Performance Shaders,
# which is a backend for matrix multiplication on Apple devices (macOS) with the M series of apple silicon.
# cuda - is refering to CUDA backend for matrix multiplication on Nvidia devices
device = "cuda"

torch.manual_seed(1337)

if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    torch.mps.manual_seed(1337)

# B could go up to 32 (and higher), but it's limited to 4 because my mac can't handle it.
train_loader = DataLoader(B=8, T=1024)
torch.set_float32_matmul_precision('high')

model = GPT(GPTConfig())
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Open a file to store the training data
with open(f'trainlog/training_log_{device}.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['step', 'loss', 'dt_ms', 'tokens_per_sec'])

    total_start_time = time.time()  # Start the timer for the entire training loop

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

        # Time difference in milliseconds
        dt = (t1 - t0) * 1000
        tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)

        # Print the data
        print(f"loss at step {i:02} | loss: {loss.item():18.15f} | dt: {dt:8.2f}ms | tok/sec: {tokens_per_sec:6.2f}")

        # Write the data to the file
        writer.writerow([i, loss.item(), dt, tokens_per_sec])

    total_end_time = time.time()  # End the timer for the entire training loop
    total_time = total_end_time - total_start_time  # Calculate the total time in seconds


print(f"Total training time: {total_time:.2f} seconds")
