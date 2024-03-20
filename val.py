import torch
from models.model.transformer import Transformer
from utils.data_loader import Load_Data, Lang
from config import *
from utils.utils import count_parameters, timeSince
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
        print('')






def run():
    # x = torch.randint(low=0, high=100, size=(1, 10), dtype=torch.int).to(device)
    # y = model(x, torch.tensor([2], device=device))
    # print("y shape", y.shape)
    QA_data = Load_Data(csv_path=csv_path, max_len=MAX_LENGTH, device = device)
    obj_lang, train_dataloader = QA_data.get_dataloader(batch_size = batch_size)
    model = Transformer(input_size = obj_lang.n_words, hidden_size=hidden_size,
                        vocab_size= obj_lang.n_words, max_len= MAX_LENGTH, device = device)
        
        
    # with mlflow.start_run() as run:
    #     mlflow.pytorch.log_model(model, "runs")
        
    # model_uri = f"runs:/{run.info.run_id}/runs"
    # loaded_model = mlflow.pytorch.load_model(model_uri)
    # loaded_model.eval()
    
    model = torch.load(PATH_SAVE)
    model.to(device)
    model.eval()
    
    evaluateRandomly(model,QA_data, n=10)
    
if __name__ == "__main__":
    run()