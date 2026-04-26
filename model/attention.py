import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.W_K = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.W_Q = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.W_V = nn.Linear(embedding_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        
        # 1. Project input through K, Q, V linear layers
        k = self.W_K(embedded)   # (B, T, attention_dim)
        q = self.W_Q(embedded)   # (B, T, attention_dim)
        v = self.W_V(embedded)   # (B, T, attention_dim)

        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        scores = q @ torch.transpose(k, 1, 2)
        context_length, attention_dim = k.shape[1], k.shape[2]
        scores = scores / (attention_dim ** 0.5)

        # 3. Apply causal mask
        mask = torch.tril(torch.ones(context_length, context_length)) == 0
        scores = scores.masked_fill(mask, float('-inf'))

        # 4. Apply softmax(dim=2) to masked scores
        weights = nn.functional.softmax(scores, dim=2)

        # 5. Return (weights @ V) rounded to 4 decimal places
        return torch.round(weights @ v, decimals=4)

