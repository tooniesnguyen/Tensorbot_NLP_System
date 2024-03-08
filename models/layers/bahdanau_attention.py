import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size, dropout = 0, **kwargs) -> None:
        super().__init__()
        self.Wq = nn.LazyLinear(hidden_size, bias = False)
        self.Wk = nn.LazyLinear(hidden_size, bias = False)
        self.Wv = nn.LazyLinear(1, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, keys, values = None):
        if values is None:
            values = keys

        scores = self.Wv(torch.tanh(self.Wq(query) + self.Wk(keys)))

        scores = scores.squeeze(2).unsqueeze(1)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, values)

        return context, weights