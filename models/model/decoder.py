import torch
from torch import nn
from models.blocks.decoder_layer import TransformerDecoderBlock
from models.embedding.positional_encoding import PositionalEncoding



class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens = 64, num_heads = 4, num_blks = 2, dropout = 0.1):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_blks = num_blks
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module("block"+str(i), TransformerDecoderBlock(num_hiddens, ffn_num_hiddens, num_heads, dropout, i))

        self.dense = nn.LazyLinear(vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens = None):
        return [enc_outputs, enc_valid_lens, [None] * self.num_blks]
    
    def forward(self, state, target_tensor=None):
        batch_size = state[0].shape[0]
        decoder_outputs = []
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        if target_tensor is not None:
            decoder_input = torch.cat((decoder_input, target_tensor), dim=1)
            decoder_outputs, _ = self.forward_step(decoder_input, state)
            # print("Decode ",torch.argmax(decoder_outputs, dim = 2))
            # print("Decode ",torch.argmax(decoder_outputs, dim = 2))

            decoder_outputs = decoder_outputs[:,:-1,:]
        else:
            for i in range(MAX_LENGTH):
                decoder_output, state = self.forward_step(decoder_input, state)
                decoder_outputs.append(decoder_output)
                _, topi = decoder_output.topk(1)
                decoder_input = topi.reshape(1,-1).detach()
            decoder_outputs = torch.cat(tuple(decoder_outputs), dim=1)
        
        # Ensure decoder_outputs is a tuple before passing it to torch.cat
        
        # print("Shape after concat ", decoder_outputs.shape)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        
        return decoder_outputs

    def forward_step(self, input_paralell, state):
        X = self.pos_encoding(self.embedding(input_paralell) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # Decoder self-attention weights
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights