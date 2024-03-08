import torch
from torch import nn
from models.blocks.encoder_layer import TransformerEncoderBlock
from models.embedding.positional_encoding import PositionalEncoding



class TransformerEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, ffn_num_hiddens = 64, num_heads = 4, num_blks =  2, dropout = 0.1, use_bias = False):
        super().__init__()
        self.num_hiddens = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.pos_encoding = PositionalEncoding(hidden_size, dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module("block"+str(i), TransformerEncoderBlock(hidden_size, ffn_num_hiddens, num_heads, dropout, use_bias))

    def forward(self, X, valid_lens = None):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X