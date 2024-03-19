import torch
from torch import nn

from models.model.encoder import TransformerEncoder
from models.model.decoder import TransformerDecoder


class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, device):
        super().__init__()
        self.encoder = TransformerEncoder(input_size, hidden_size).to(device)
        self.decoder = TransformerDecoder(vocab_size, hidden_size).to(device)
    def forward(self, src, trg = None):
        encoder_src = self.encoder(src)
        decoder_state = self.decoder.init_state(encoder_src)
        output = self.decoder(decoder_state, trg)
        return output
    
        