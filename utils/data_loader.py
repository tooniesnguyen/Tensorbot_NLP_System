import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from io import open
import unicodedata
import re
import random

from .utils import *
import sys
import socket
import os
from pathlib import Path
import json


class Load_Data:
    def __init__(self, data_path, max_len = 10, device = "CPU", type_data = "json"):
        self.device = device
        self.max_len = max_len
        if type_data == "csv":
            self.df = pd.read_csv(data_path, sep = ",", header = None, skiprows=1)
            self.df_filter = self.df.applymap(self._norm_string).values.tolist()
        else:
            self.df_filter = self.read_json(data_path)
        self.object_lang, self.pairs = self.PrepareData()
        
    def _filter_pair(self, pair):
        return len(pair[0].split(' ')) < self.max_len and \
            len(pair[1].split(' ')) < self.max_len 
    def _filter_pairs(self, pairs):
        return [pair for pair in pairs if self._filter_pair(pair)]

    def PrepareData(self, name = "English"):
        '''
        Input must be dataframe fillter just two column Q&A
        '''
        _pairs, _object_lang = self.df_filter, Lang(name)
        print("Read %s sentence pairs" % len(_pairs))
        _pairs = self._filter_pairs(_pairs)
        print("Trimmed to %s sentence pairs" % len(_pairs))
        print("Counting words...")
        for pair in _pairs:
            _object_lang.add_sentence(pair[0])
            _object_lang.add_sentence(pair[1])
            
        print("Counted words: ")
        print("Total bag of word: ", _object_lang.n_words)
        return _object_lang, _pairs
    
    def indexesFromSentence(self, sentence):
        return [self.object_lang.word2index.get(word, self.object_lang.word2index["#UNK"]) for word in sentence.split(" ")]

    def tensorFromSentence(self, sentence):
        _indexes = self.indexesFromSentence(sentence)
        _indexes.append(self.object_lang.word2index["#END"])
        return torch.tensor(_indexes, dtype = torch.long, device = self.device).reshape(1,-1)
    def get_dataloader(self, batch_size):
        n = len(self.pairs)
        source_ids = np.ones((n, self.max_len), dtype = np.int32)*self.object_lang.word2index["#PAD"]
        src_valid_len = np.ones((n), dtype = np.int32)
        target_ids = np.ones((n, self.max_len), dtype = np.int32)*self.object_lang.word2index["#PAD"]

        for idx, (src, tgt) in enumerate(self.pairs):
            src_ids = self.indexesFromSentence(src)
            tgt_ids = self.indexesFromSentence(tgt)
            
            src_ids.append(self.object_lang.word2index["#END"])
            tgt_ids.append(self.object_lang.word2index["#END"])

            src_valid_len[idx] = len(src_ids)
            source_ids[idx, :len(src_ids)] = src_ids
            target_ids[idx, :len(tgt_ids)] = tgt_ids

        train_data = TensorDataset(torch.LongTensor(source_ids).to(self.device),
                                   torch.LongTensor(src_valid_len).to(self.device),
                                   torch.LongTensor(target_ids).to(self.device))
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
        return self.object_lang, train_dataloader
    
    def read_json(self, json_dir):
        f = open(json_dir)
        json_data = json.load(f)
        json_data = json_data["data"]

        train_data = []
        for item in json_data:
            for paragraph in item["paragraphs"]:
                for qa in paragraph["qas"]:
                    question = qa["question"]
                    answers = [answer["text"] for answer in qa["answers"]]
                    if len(answers) == 0:
                        answers = ["I am impossible answer this question"]
                    train_data.append([question, answers[0]])
        f.close()
        return train_data

@add_func_class(Load_Data)
def _norm_string(self, s):
    def _Unicode2Ascii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn')
    s = _Unicode2Ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()


class Lang:
    def __init__(self, name):
        '''
        #BEG: Begin of Sentence
        #END: End of Sentence
        #PAD: Padding
        #UNK: Unknow
        '''
        self.name = name
        self.word2index = {"#BEG": 0, "#END": 1, "#PAD": 2, "#UNK": 3}
        self.index2word = {0: "#BEG", 1: "#END", 2: "#PAD", 3: "#UNK"}
        self.word2count = {}
        self.n_words = 4 
        
    def add_sentence(self, sentence):
        for word in sentence.split(" "):
            self._add_word(word)
            
    def _add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1



if __name__ == "__main__":
    HOST = socket.gethostbyname(socket.gethostname())
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]
    WORK_DIR = os.path.dirname(ROOT)
    
    DF_DIR = f"{WORK_DIR}/data/AI.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(next(iter(Load_Data(DF_DIR).get_dataloader(batch_size=32)[1])))
