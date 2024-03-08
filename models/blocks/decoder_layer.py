import torch.nn as nn
import torch
import torch.nn.functional as F
from layers.bahdanau_attention import BahdanauAttention


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size, dropout= dropout_p)
        # batch_first=True 
        # ->input và output sẽ có dạng (batch_size, seq_len, features) thay vì (seq_len, batch_size, features) như mặc định
        self.gru = nn.GRU(2*hidden_size, hidden_size, batch_first=True) 
        self.out = nn.LazyLinear(output_size)

        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input  = torch.empty(batch_size, 1, dtype = torch.long, device = device).fill_(SOS_token)
        # print("Hidden encoder shape", encoder_hidden.shape)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, atten_weight = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)

            decoder_outputs.append(decoder_output)
            attentions.append(atten_weight)

            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)

            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach() 
                
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))

        query = hidden.permute(1, 0 ,2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru  = torch.cat((embedded, context), dim = 2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)
        return output, hidden, attn_weights

