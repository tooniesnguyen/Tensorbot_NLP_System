import torch
from models.model.transformer import Transformer
from utils.data_loader import Load_Data, Lang
from config import *
from utils.utils import count_parameters
import torch.nn as nn
from torch import optim

import time
import math




# count_parameters(model)


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def train_epoch(dataloader, model, model_optimizer, criterion):
    total_loss = 0
    for data in dataloader:
        src_tensor, valid_len_src, trg_tensor = data
        
        model_optimizer.zero_grad()
        decoder_outputs = model(src_tensor, valid_len_src, trg_tensor)
        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            trg_tensor.view(-1)
        )
        loss.backward()
        
        model_optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss/len(dataloader)


def train(train_dataloader, model, n_epochs, learning_rate=0.001,
               print_every=10, plot_every=100):
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0 
    
    model_optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    criterion = nn.NLLLoss()
    
    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, model, model_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss
        
        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                        epoch, epoch / n_epochs * 100, print_loss_avg))

def run():
    # x = torch.randint(low=0, high=100, size=(1, 10), dtype=torch.int).to(device)
    # y = model(x, torch.tensor([2], device=device))
    # print("y shape", y.shape)
    QA_data = Load_Data(csv_path=csv_path, max_len=MAX_LENGTH, device = device)
    obj_lang, train_dataloader = QA_data.get_dataloader(batch_size = batch_size)
    model = Transformer(input_size = obj_lang.n_words, hidden_size=hidden_size,
                        vocab_size= obj_lang.n_words, max_len= MAX_LENGTH, device = device)
    
    train(train_dataloader, model, 20)
    
    
if __name__ == "__main__":
    run()