import torch
from torch import nn
from models.layers.attention import BahdanauAttention



class MultiHeadAttention(nn.Module):
    '''
    Input: X = (batch_size, num_queries, num_hiddens)
           Y = (batch_size, num_kvpairs, num_hiddens)
    
    Output: (batch_size, num_queries, num_hiddens)
    
    '''
    """Multi-head attention."""
    def __init__(self, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.attention = BahdanauAttention(num_hiddens)
        self.W_q = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.W_k = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.W_v = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        queries = self.transpose_qkv(self.W_q(queries))

        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))

        if valid_lens is not None:
            # On axis 0, copy the first item (scalar or vector) for num_heads
            # times, then copy the next item, and so on
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # Shape of output: (batch_size * num_heads, no. of queries,
        # num_hiddens / num_heads)
        output = self.attention(queries, keys, values, valid_lens)
        # Shape of output_concat: (batch_size, no. of queries, num_hiddens)
        
        output_concat = self.transpose_output(output)
        return self.W_o(output_concat)
    def transpose_qkv(self, X):
        """Transposition for parallel computation of multiple attention heads."""

        X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)

        X = X.permute(0, 2, 1, 3)

        return X.reshape(-1, X.shape[2], X.shape[3])
    def transpose_output(self, X):
        """Reverse the operation of transpose_qkv."""
        X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)