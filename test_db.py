import torch
import torch.nn as nn
from torch.nn import functional as F
import sqlite3
import re
import sentencepiece as spm
import tiktoken

torch.manual_seed(1337)

device = torch.device("mps")

conn = sqlite3.connect('smaller.db')
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
cursor.execute("SELECT * FROM data")  # Replace with the correct table name
text = cursor.fetchall()

conn.close()
text_corpus = ""
for tup in text:
    sentence = tup[1]
    text_corpus += sentence + " "

with open('text_data.txt', 'w', encoding='utf-8') as f:
    f.write(text_corpus)
text_corpus = text_corpus.replace("\n", "<NEWLINE>")
print(text_corpus)
spm.SentencePieceTrainer.Train('--input=text_data.txt --model_prefix=m --vocab_size=20000 --character_coverage=1.0 --control_symbols=<NEWLINE>')
sp = spm.SentencePieceProcessor(model_file='m.model')

encoded_corpus = sp.encode(text_corpus, out_type=int)
encoded_tensor = torch.tensor(encoded_corpus, dtype=torch.long)
print("Encoded Tensor:", encoded_tensor)
print(sp.vocab_size())
newline_token_id = sp.encode('<s>', out_type=int)

print(f"Token ID for newline character: {newline_token_id}")
print(sp.decode([0]))

encoding = tiktoken.encoding_for_model("gpt-4o-mini")

print(torch.tensor(encoding.encode("\n"), dtype=torch.long, device=device))
print(encoding.n_vocab)