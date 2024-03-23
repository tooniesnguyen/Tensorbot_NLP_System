import torch
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

import mlflow
import mlflow.pytorch

mlflow.pytorch.autolog()




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
        print("Bredict Word tokenize ", output_sentence[:-5])
        print("Target  word tokenize ", pair[1])
        print("Bleu score ", calc_bleu(output_sentence[:-5], pair[1]))
        print('')






def run():
    # x = torch.randint(low=0, high=100, size=(1, 10), dtype=torch.int).to(device)
    # y = model(x, torch.tensor([2], device=device))
    # print("y shape", y.shape)
    QA_data = Load_Data(data_path=csv_path,save_dict=False, dict_path = dict_path , mode_load="train", type_data="csv", max_len=MAX_LENGTH, device = device)
    obj_lang, train_dataloader = QA_data.get_dataloader(batch_size = batch_size)
    model = Transformer(input_size = obj_lang.n_words, hidden_size=hidden_size,
                        vocab_size= obj_lang.n_words, max_len= MAX_LENGTH, device = device)
        
    model = torch.load(PATH_SAVE)
    model.to(device)
    model.eval()
    
    evaluateRandomly(model,QA_data, n=10)
    
if __name__ == "__main__":
    run()