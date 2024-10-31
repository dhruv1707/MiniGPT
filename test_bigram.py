import torch
import torch.nn as nn

class MultiHead(nn.Module):
    "multi head attention using parallel processing across heads"

    def __init__(self, num_heads, head_size, n_embd, block_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.key = nn.Linear(n_embd, num_heads * head_size, bias=False)
        self.query = nn.Linear(n_embd, num_heads * head_size, bias=False)
        self.value = nn.Linear(n_embd, num_heads * head_size, bias=False)
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)  # (B, num_heads, T, head_size)
        q = self.query(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)  # (B, num_heads, T, head_size)
        v = self.value(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)  # (B, num_heads, T, head_size)

        wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)  # (B, num_heads, T, T)

        wei = wei.masked_fill(self.tril[:, :, :T, :T] == 0, float("-inf"))  # (B, num_heads, T, T)
        wei = torch.softmax(wei, dim=-1)  # (B, num_heads, T, T)
        out = wei @ v  # (B, num_heads, T, head_size)

        out = out.view(B, T, self.num_heads * self.head_size)

        out = self.proj(out)

        return out

class FeedForward(nn.Module):
    "a simple linear layer followed by non-linearity"

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd*4),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd)
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    "Transformer block: communication followed by computation"

    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        head_size = n_embd//n_head
        self.sa = MultiHead(num_heads=n_head, head_size=head_size, n_embd=n_embd, block_size=block_size)
        self.ff = FeedForward(n_embd=n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        print(f"Output shape after self attention: {x.shape}")
        x = x + self.ff(self.ln2(x))
        print(f"Output shape after feed forward: {x.shape}")
        return x


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Initialize parameters
    B = 2  # Batch size
    T = 8  # Sequence length
    n_embd = 288  # Embedding dimension
    num_heads = 6  # Number of heads
    block_size = T  # Adjust as needed

    # Create sample input
    x = torch.randn(B, T, n_embd)

    # Create the multi-head attention module


    block = Block(n_embd=n_embd, n_head=num_heads, block_size=T)

    # Forward pass
    output = block(x)

    # Print the output shape
    print("Output shape from MultiHead:", output.shape)  # Should be (B, T, n_embd)

    # Optionally print sample output values
    print("Sample output values:", output)