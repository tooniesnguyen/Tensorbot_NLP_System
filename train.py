import torch
from models.model.transformer import Transformer
from utils.data_loader import Load_Data, Lang
from config import *
from utils.utils import count_parameters, timeSince
import torch.nn as nn
from torch import optim
import time
import math
import shutil
import os
import random
import warnings 
warnings.filterwarnings('ignore') 

import mlflow
import mlflow.pytorch
from mlflow.models import infer_signature

mlflow.pytorch.autolog()


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
            mlflow.log_metric("val_loss", print_loss_avg)
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                        epoch, epoch / n_epochs * 100, print_loss_avg))
def evaluate(model, sentence, obj_data):
    with torch.no_grad():
        input_tensor = obj_data.tensorFromSentence(sentence)

        decoder_outputs= model(input_tensor)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(obj_data.object_lang.index2word[idx.item()])
    return decoded_words


def evaluateRandomly(model,obj_data, n=10):
    pairs = obj_data.pairs
    print(pairs[0][0])
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words = evaluate(model, pair[0], obj_data)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


def run():
    # x = torch.randint(low=0, high=100, size=(1, 10), dtype=torch.int).to(device)
    # y = model(x, torch.tensor([2], device=device))
    # print("y shape", y.shape)
    QA_data = Load_Data(csv_path=csv_path, max_len=MAX_LENGTH, device = device)
    obj_lang, train_dataloader = QA_data.get_dataloader(batch_size = batch_size)
    model = Transformer(input_size = obj_lang.n_words, hidden_size=hidden_size,
                        vocab_size= obj_lang.n_words, max_len= MAX_LENGTH, device = device)
    
    train(train_dataloader, model, 100, print_every= 10)
    
    
    if os.path.exists(path):
        shutil.rmtree(path)
    mlflow.pytorch.save_model(model, path)
    
    torch.save(model, PATH_SAVE)
    
    # model.eval()
    # evaluateRandomly(model, QA_data)

    
if __name__ == "__main__":
    run()