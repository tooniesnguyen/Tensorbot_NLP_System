import torch
from torch import nn

from models.model.encoder import TransformerEncoder
from models.model.decoder import TransformerDecoder


class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, device):
        super().__init__()
        self.encoder = TransformerEncoder(input_size, hidden_size).to(device)
        self.decoder = TransformerDecoder(vocab_size, hidden_size).to(device)
    def forward(self, src, trg):
        encoder_src = self.encoder(src)
        decpder_state = self.decoder.init_state(enc_outputs)
        output = self.decoder(dec_state, trg)
        return output
    
        