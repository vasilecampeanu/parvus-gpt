import tiktoken
import torch

class DataLoader:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # At init load tokens from disk and store them in memory
        with open('../datasets/tinyshakespeare.txt', 'r') as f:
            text = f.read()

        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)

        print(f"Loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # State
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T

        # +1 represents the target token
        buf = self.tokens[self.current_position:self.current_position+B*T+1]
        
        # Inputs
        x = (buf[:-1]).view(B, T)

        # Targets
        y = (buf[1:]).view(B, T)

        # Advance the position in the tensor
        self.current_position += B * T

        # If loading the next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0

        return x, y
