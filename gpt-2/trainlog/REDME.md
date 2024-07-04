# Running on MacOS

This is the configuartion we had.

```
train_loader = DataLoader(B=8, T=1024)
```

## Unoptimized

### CPU

Total training time: `484.71 seconds`

### MPS

We have a problem on MacOS. Train loss is not converging when using device MPS for some strange reason.

#### Trains

- 01 Total training time: 118.84 seconds
- 02 Total training time: 118.90 seconds
- 03 Total training time: 107.35 seconds
- 04 Total training time: 100.84 seconds
- 05 Total training time: 139.70 seconds

The average train time on MPS: is 116 seconds

# Running on PodRun 1 x RTX 4090 | 32 vCPU 125 GB VRAM

## Unoptimized

### CUDA with RTX 4090 24GB VRAM

Total training time: 13.96 seconds

### CUDA with A100 SMX 80GB VRAM

Total training time: 26.22 seconds
