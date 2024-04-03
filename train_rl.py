import torch
import torch.nn.functional as F
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
import numpy as np
import warnings 
warnings.filterwarnings('ignore') 
torch.jit.enable_onednn_fusion(True)
torch.backends.cudnn.benchmark = True

import mlflow
import mlflow.pytorch
from mlflow.models import infer_signature

mlflow.pytorch.autolog()


def decoder_word(index_sentence, object_lang):
    decoded_words = []
    for idx in index_sentence:
        if idx.item() == EOS_token:
            # decoded_words.append('<EOS>')
            break
        decoded_words.append(object_lang.index2word[idx.item()])
    decoded_sentence = ' '.join(decoded_words)
    decoded_sentence = decoded_sentence
    return decoded_sentence

def evaluateRandomly(model,obj_data, n=10):
    '''
    Input: [["Ques_1", "Answer 1"], ["Ques_2", "Answer 2"], ...]
    Output: [["Predict_1", "Answer 1"], ["Predict_2", "Answer 2"], ...]
    
    '''
    __pairs_random = random.sample(obj_data.pairs, n)
    __bleu_mean = 0
    for __pair in __pairs_random:
        
        __pair_src_tensor = obj_data.tensorFromSentence(__pair[0])
        __decoder_outputs= F.log_softmax(model(__pair_src_tensor), dim=-1)    
        
        _, __topi = __decoder_outputs.topk(1)
        __decoded_ids = __topi.squeeze()
        __decoded_id2word = decoder_word(__decoded_ids, obj_data.object_lang)
        
        __bleu_mean += calc_bleu(__decoded_id2word,  __pair[1])
    
    # print("------------------------------------------------------------------------------")
    # print("Reference text: ", __pair[1])        
    # print("Predict text: ",__decoded_id2word)
    # print("Bleu return ", __bleu_mean/n)
    # print("------------------------------------------------------------------------------")    
    return __bleu_mean/n

def train_epoch(dataloader,val_pairs,object_lang , model, model_optimizer, criterion):
    total_loss = 0
    total_bleu_argmax = 0
    
    total_bleu_valid = 0
    
    bleus_argmax = []
    bleus_sample = []
    for data in dataloader:
        src_tensor, valid_len_src, trg_tensor = data
        model_optimizer.zero_grad()
        decoder_outputs = model(src_tensor, valid_len_src, trg_tensor) # output (32, 10, 488) with 488 is bag of words
        
        ################################ BLEU of Train  Argmax #####################################
        net_policies = []
        net_actions = []
        net_advantages = []
        
        for idx, decoder_output in enumerate(decoder_outputs):
            r_outputs = torch.clone(decoder_output)
            _, topi = decoder_output.topk(1)
            decoded_ids = topi.squeeze()

            actions = decoder_word(decoded_ids, object_lang)
            ref_indices = decoder_word(trg_tensor[idx], object_lang)
            argmax_bleu = calc_bleu(actions, ref_indices)
            bleus_argmax.append(argmax_bleu)
            
            if argmax_bleu > 0.99:
                continue
        #############################################################################################
        
        ################################ BLEU of Train  Sample #####################################
            for _ in range(num_samples):
                decoder_output_sm = F.softmax(decoder_output, dim = -1)
                action_sample = torch.multinomial(decoder_output_sm, 1, replacement=True)
                decoded_samples_ids = action_sample.squeeze()
                action_decoder = decoder_word(decoded_samples_ids, object_lang)
                sample_bleu = calc_bleu(action_decoder, ref_indices)
                net_policies.append(r_outputs)
                net_actions.extend(action_sample)
            
                adv = sample_bleu - argmax_bleu
                net_advantages.extend([adv]*len(action_sample))
                bleus_sample.append(sample_bleu)

        #############################################################################################
        
        ################################ Calculate BLEU of Valid ####################################
        bleu_mean_valid = evaluateRandomly(model,val_pairs)
        total_bleu_valid += bleu_mean_valid
        #############################################################################################

        if not net_policies:
            continue
        
        policies_v = torch.cat(net_policies)
        actions_t = torch.LongTensor(net_actions).to(device)
        adv_v = torch.FloatTensor(net_advantages).to(device)
        log_prob_v = F.log_softmax(policies_v, dim=1)
        lp_a = log_prob_v[range(len(net_actions)),actions_t]
        log_prob_actions_v = adv_v * lp_a
        loss_policy_v = -log_prob_actions_v.mean()
        loss_v = loss_policy_v
        loss_v.backward()
        model_optimizer.step()
        
        total_loss += loss_v.item()
        
        # print(bleus_argmax)
    return total_loss/len(dataloader), sum(bleus_argmax)/len(bleus_argmax), total_bleu_valid/len(dataloader)




def train(train_dataloader, val_pairs, object_lang, model, n_epochs, learning_rate= 1e-4,
               print_every=10, save_epoch=100):
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    print_bleu_total_argmax = 0 
    print_bleu_valid_total = 0
    model_optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    criterion = nn.NLLLoss()
    
    for epoch in range(1, n_epochs + 1):
        loss, bleu_score_argmax, bleu_score_valid = train_epoch(dataloader=train_dataloader, val_pairs = val_pairs,object_lang = object_lang, model = model,
                           model_optimizer = model_optimizer,criterion= criterion)
        print_loss_total += loss
        print_bleu_total_argmax += bleu_score_argmax
        print_bleu_valid_total += bleu_score_valid
        
        
        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_bleu_avg_argmax = print_bleu_total_argmax/ print_every
            print_bleu_valid_avg = print_bleu_valid_total/print_every
            
            ######################### PLOT MLFLOW ##########################
            mlflow.log_metric("train_loss", print_loss_avg)
            mlflow.log_metric("train_bleu_score", print_bleu_avg_argmax)
            
            mlflow.log_metric("valid_bleu_score", print_bleu_valid_avg)
            ################################################################
            print_bleu_total_argmax = 0 
            print_loss_total = 0
            print_bleu_valid_total = 0
            
            print('%s (%d %d%%) loss %.4f  bleu_score %.4f, bleu_score_valid %.4f' % (timeSince(start, epoch / n_epochs),
                                        epoch, epoch / n_epochs * 100, print_loss_avg, print_bleu_avg_argmax, print_bleu_valid_avg))
            
            if epoch % save_epoch == 0:
            ###################### SAVE MODEL ##########################
                os.makedirs(f"{PATH_SAVE_RL}", exist_ok=True)  
                name_file_pth = f"{PATH_SAVE_RL}/epoch{str(epoch)}.pth"
                torch.save(model, name_file_pth)
                print("Saved model successfull")
            ############################################################

def run():
    QA_data = Load_Data(data_path=json_path_train,save_dict=True, dict_path = dict_path_json , mode_load="train", 
                        type_data="json", max_len=MAX_LENGTH, device = device)
    obj_lang, train_dataloader = QA_data.get_dataloader(batch_size = batch_size)
    print("Use device ", device)

    mlflow.log_param("lr", learning_rate)
    mlflow.log_param("epoch_th", epoch_th)
    path_save = os.path.join(f"{PATH_SAVE}", f"epoch{epoch_th}.pth")
    model = torch.load(path_save).to(device)
    valid_data = Load_Data(data_path=json_path_dev,save_dict=False, dict_path = dict_path_json, mode_load="train",
                           type_data="json", max_len=MAX_LENGTH, device = device)
    train(train_dataloader, val_pairs=valid_data  ,object_lang = obj_lang ,model = model,n_epochs = 2000, print_every= 1)
    
if __name__ == "__main__":
    run()