import torch
import torch.nn.functional as F
from models.model.transformer import Transformer
from utils.data_loader import Load_Data, Lang
from config import *
from utils.utils import count_parameters, timeSince, calc_bleu_many, bleu, calc_bleu
from nltk.tokenize import word_tokenize
import torch.nn as nn
from torch import optim
import time
import math
import random
import numpy as np
import mlflow
import mlflow.pytorch

mlflow.pytorch.autolog()



def evaluate(model, sentence, obj_data):
    with torch.no_grad():
        input_tensor = obj_data.tensorFromSentence(sentence)

        decoder_outputs= F.log_softmax(model(input_tensor),dim = -1)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                # decoded_words.append('<EOS>')
                break
            decoded_words.append(obj_data.object_lang.index2word[idx.item()])
    return decoded_words


def evaluateRandomly(model,obj_data, n=10, random_state = 23):
    random.seed(random_state)
    pairs = obj_data.pairs
    for i in range(n):
        pair = random.choice(pairs)
        
        output_words = evaluate(model, pair[0], obj_data)
        output_sentence = ' '.join(output_words)
        if calc_bleu(output_sentence, pair[1]) < 1:
            print('>', pair[0])
            print('=', pair[1])  
            print('<', output_sentence)
            # print("Bredict Word tokenize :", output_sentence)
            # print("Target  word tokenize :", pair[1])
            print("Bleu score ", calc_bleu(output_sentence, pair[1]))
            print('-----------------------------------------')






def run():
    # x = torch.randint(low=0, high=100, size=(1, 10), dtype=torch.int).to(device)
    # y = model(x, torch.tensor([2], device=device))
    # print("y shape", y.shape)
    QA_data = Load_Data(data_path=json_path_train,save_dict=False, dict_path = dict_path_json , mode_load="train", type_data="json", max_len=MAX_LENGTH, device = device)
    obj_lang, train_dataloader = QA_data.get_dataloader(batch_size = batch_size)
    model = Transformer(input_size = obj_lang.n_words, hidden_size=hidden_size,
                        vocab_size= obj_lang.n_words, max_len= MAX_LENGTH, device = device)
        
    model = torch.load(f"{PATH_SAVE}/pretrain/epoch100.pth")
    model = model.to(device)
    model.eval()
    
    evaluateRandomly(model,QA_data, n=100)
    
if __name__ == "__main__":
    run()