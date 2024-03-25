import torch
from models.model.transformer import Transformer
from utils.data_loader import Load_Data, Lang
from config import *
from utils.utils import count_parameters, timeSince, calc_bleu_many, calc_bleu
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



def decoder_word(index_sentence, object_lang):
    decoded_words = []
    for idx in index_sentence:
        if idx.item() == EOS_token:
            decoded_words.append('<EOS>')
            break
        decoded_words.append(object_lang.index2word[idx.item()])
    decoded_sentence = ' '.join(decoded_words)
    decoded_sentence = decoded_sentence[:-5]
    return decoded_sentence



def evaluate(model, sentence, obj_data):
    input_tensor = obj_data.tensorFromSentence(sentence)
    decoder_outputs= model(input_tensor)
    _, topi = decoder_outputs.topk(1)
    decoded_ids = topi.squeeze()
    


def evaluateRandomly(model,obj_data, n=10):
    '''
    Input: [["Ques_1", "Answer 1"], ["Ques_2", "Answer 2"], ...]
    Output: [["Predict_1", "Answer 1"], ["Predict_2", "Answer 2"], ...]
    
    '''
    __pairs_random = random.sample(obj_data.pairs, n)
    __bleu_mean = 0
    for __pair in __pairs_random:
        
        __pair_src_tensor = obj_data.tensorFromSentence(__pair[0])
        __decoder_outputs= model(__pair_src_tensor)        
        
        _, __topi = __decoder_outputs.topk(1)
        __decoded_ids = __topi.squeeze()
        __decoded_id2word = decoder_word(__decoded_ids, obj_data.object_lang)
        
        __bleu_mean += calc_bleu(__decoded_id2word,  __pair[1])
        
    # print("Predict text: ",__decoded_id2word)
    # print("Reference text: ", __pair[1])
    # print("Bleu return ", __bleu_mean/n)
    return __bleu_mean/n


def train_epoch(dataloader,val_pairs,object_lang , model, model_optimizer, criterion):
    total_loss = 0
    total_bleu = 0
    
    total_bleu_valid = 0
    for data in dataloader:
        src_tensor, valid_len_src, trg_tensor = data
        
        model_optimizer.zero_grad()
        decoder_outputs = model(src_tensor, valid_len_src, trg_tensor)
        
        ################################ Calculate BLEU of Train ####################################
        
        _, topi = decoder_outputs[-1].topk(1)
        decoded_ids = topi.squeeze()

        output_decoded = decoder_word(decoded_ids, object_lang)
        trg_decoded = decoder_word(trg_tensor[-1], object_lang)
        # print("bleu score ", calc_bleu(output_decoded, trg_decoded))
        
        total_bleu += calc_bleu(output_decoded, trg_decoded)
        #############################################################################################
        
        
        
        ################################ Calculate BLEU of Valid ####################################
        bleu_mean_valid = evaluateRandomly(model,val_pairs)
        total_bleu_valid += bleu_mean_valid
        #############################################################################################

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            trg_tensor.view(-1)
        )
        loss.backward()
        
        model_optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss/len(dataloader), total_bleu/len(dataloader), total_bleu_valid/len(dataloader)




def train(train_dataloader, val_pairs, object_lang, model, n_epochs, learning_rate=0.001,
               print_every=10):
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    print_bleu_total = 0 
    print_bleu_valid_total = 0
    model_optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    criterion = nn.NLLLoss()
    
    for epoch in range(1, n_epochs + 1):
        loss, bleu_score, bleu_score_valid = train_epoch(dataloader=train_dataloader, val_pairs = val_pairs,object_lang = object_lang, model = model,
                           model_optimizer = model_optimizer,criterion= criterion)
        
        print_loss_total += loss
        print_bleu_total += bleu_score
        print_bleu_valid_total += bleu_score_valid
        
        
        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_bleu_avg = print_bleu_total/ print_every
            print_bleu_valid_avg = print_bleu_valid_total/print_every
            
            ######################### PLOT MLFLOW ##########################
            mlflow.log_metric("train_loss", print_loss_avg)
            mlflow.log_metric("train_bleu_score", print_bleu_avg)
            
            mlflow.log_metric("valid_bleu_score", print_bleu_valid_avg)
            ################################################################
            print_bleu_total = 0 
            print_loss_total = 0
            print_bleu_valid_total = 0
            
            print('%s (%d %d%%) loss %.4f  bleu_score %.4f, bleu_score_valid %.4f' % (timeSince(start, epoch / n_epochs),
                                        epoch, epoch / n_epochs * 100, print_loss_avg, print_bleu_avg, print_bleu_valid_avg))
            
            
            ###################### SAVE MODEL ##########################
            name_file_pth = f"{PATH_SAVE}/epoch{str(epoch)}.pth"
            torch.save(model, name_file_pth)
            print("Saved model successfull")
            ############################################################

def run():

    QA_data = Load_Data(data_path=json_path_train,save_dict=True, dict_path = dict_path , mode_load="train", 
                        type_data="json", max_len=MAX_LENGTH, device = device)
    obj_lang, train_dataloader = QA_data.get_dataloader(batch_size = batch_size)
    model = Transformer(input_size = obj_lang.n_words, hidden_size=hidden_size,
                        vocab_size= obj_lang.n_words, max_len= MAX_LENGTH, device = device)
    
    
    valid_data = Load_Data(data_path=json_path_dev,save_dict=False, dict_path = dict_path, mode_load="train",
                           type_data="json", max_len=MAX_LENGTH, device = device)
    train(train_dataloader, val_pairs=valid_data  ,object_lang = obj_lang ,model = model,n_epochs = 1000, print_every= 10)
    
    
    
if __name__ == "__main__":
    run()