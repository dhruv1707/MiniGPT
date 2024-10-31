import torch
import torch.nn as nn
from torch.nn import functional as F
import sqlite3
import re
import sentencepiece as spm


# We always start with a dataset to train on. Let's download the tiny shakespeare dataset
#wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt


batch_size = 64
block_size = 100
max_iters = 2000
eval_interval = 400
learning_rate = 1e-5
device = torch.device("mps")
eval_iters = 300
n_embd = 768
n_head = 16
n_layer = 8
dropout = 0.4

torch.manual_seed(1337)
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

conn = sqlite3.connect('smaller.db')
cursor = conn.cursor()
cursor.execute("SELECT * FROM data")  # Replace with the correct table name
text = cursor.fetchall()

conn.close()
text_corpus = ""
words = []
for tup in text:
    sentence = tup[1]
    text_corpus += sentence + " "
    

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(words) }
itos = { i:ch for i,ch in enumerate(words) }
#encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
#decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
def encode(text):
    encoded = []
    for word in text:
        word = word.lower()
        if word in stoi:
            encoded.append(stoi[word])
        else:
            print(f"Warning: '{word}' not found in vocabulary.")  # Debugging statement
            encoded.append(-1)  # You can assign a special token ID for unknown words
    return encoded
decode = lambda l: ' '.join([itos[i] for i in l if i in itos]) # decoder: take a list of integers, output a string

with open('text_data.txt', 'w', encoding='utf-8') as f:
    f.write(text_corpus)

spm.SentencePieceTrainer.Train('--input=text_data.txt --model_prefix=m --vocab_size=20000')
sp = spm.SentencePieceProcessor(model_file='m.model')

encoded_corpus = sp.encode(text_corpus, out_type=int)
data = torch.tensor(encoded_corpus, dtype=torch.long)
vocab_size = sp.vocab_size()
print(vocab_size)

# Train and test splits
#data = torch.tensor(encode(words), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

    
class MultiHead(nn.Module):
    "multi head attention using parallel processing across heads"

    def __init__(self, n_embd):
        super().__init__()
        self.attn = nn.Linear(n_embd, n_embd*3, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, x):
        B, T, C = x.shape
        q,k,v = self.attn(x).split(n_embd, dim=2)
        k = k.view(B, T, n_head, n_embd//n_head).transpose(1,2) # (B, num_heads, T, head_size)
        q = q.view(B, T, n_head, n_embd//n_head).transpose(1,2) # (B, num_heads, T, head_size)
        v = v.view(B, T, n_head, n_embd//n_head).transpose(1,2) # (B, num_heads, T, head_size)

        wei = q @ k.transpose(-2, -1)*((n_embd//n_head)**-0.5) # (B, num_heads, T, T)

        wei = wei.masked_fill(self.tril[:, : , :T, :T] == 0, float("-inf")) # (B, num_heads, T, T)
        wei = F.softmax(wei, dim=-1) # (B, num_heads, T, T)
        wei = self.dropout(wei)
        out = wei @ v # (B, num_heads, T, head_size)

        out = out.transpose(1,2).contiguous().view(B,T,C)

        out = self.dropout(self.proj(out))
        
        return out

class FeedForward(nn.Module):
    "a simple linear layer followed by non-linearity"

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd*4),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    "Transformer block: communication followed by computation"

    def __init__(self, n_embd):
        super().__init__()
        self.sa = MultiHead(n_embd=n_embd)
        self.ff = FeedForward(n_embd=n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x
    
class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd=n_embd) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature, top_k):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            logits = logits / temperature
            if top_k is not None:
                top_k_values, top_k_indices = torch.topk(logits, k=top_k, dim=-1)
                logits = torch.full_like(logits, float('-inf'))  # Set all logits to -inf
                logits.scatter_(1, top_k_indices, top_k_values)  # Fill in the top-k logits
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
model = GPTLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


for iter in range(max_iters):
    print(f"Iteration {iter}")
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.tensor([[sp.encode("_")[0]]], dtype=torch.long, device=device)

print(sp.decode(m.generate(context, max_new_tokens=500, temperature=0.8, top_k=100)[0].tolist()))
