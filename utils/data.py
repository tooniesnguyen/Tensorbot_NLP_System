'''
- The first filter limits the maximum number of tokens in training pairs (20 word)
- The second filter is reducing the number of words in our dictionary, we can reduce memory and improve the training speed. ( appear freq <= 10 ~ removed)

'''
import os
import logging
from . import utils
from . import cornell
import sys
import collections
import itertools
import pickle
# Create log with "cornell" name
log = logging.getLogger("cornell")

DATA_DIR = "data/cornell"
MAX_TOKENS = 20
MIN_TOKEN_FEQ = 10
UNKNOWN_TOKEN = '#UNK'
BEGIN_TOKEN = "#BEG"
END_TOKEN = "#END"
    
SHUFFLE_SEED = 5871

EMB_DICT_NAME = "emb_dict.dat"
EMB_NAME = "emb.npy"


def dialogues_to_pairs(dialogues, max_tokens = None): # 
    """
    Convert dialogues to training pairs of phrases

    Convet thành các cặp câu để  train

    :param dialogues:
    :param max_tokens: limit of tokens in both question and reply
    :return: list of (phrase, phrase) pairs
    """
    result = []
    for dial in dialogues: # dial: [["he's", 'not', 'in', 'this', 'building', '.'], ['all', 'right', ',', 'where', 'is', 'he', '?']]
        prev_phrase = None
        for phrase in dial: # phrase: ["he's", 'not', 'in', 'this', 'building', '.']
            if prev_phrase is not None:
                if max_tokens is None or (len(prev_phrase) <= max_tokens and len(phrase) <= max_tokens):
                    result.append((prev_phrase, phrase))
            prev_phrase = phrase

    return result

            

def load_data(genre_filter, max_tokens=MAX_TOKENS, min_token_freq=MIN_TOKEN_FEQ): #
    # [["he's", 'not', 'in', 'this', 'building', '.'], ['all', 'right', ',', 'where', 'is', 'he', '?'], [.....]]
    dialogues = cornell.load_dialogues(genre_filter=genre_filter)

    if not dialogues:
        log.error("No dialogues found, exit!")
        sys.exit()
    log.info("Loaded %d dialogues with %d phrases, generating training pairs",
             len(dialogues), sum(map(len, dialogues)))
    phrase_pairs = dialogues_to_pairs(dialogues, max_tokens=max_tokens) # [(["he's", 'not', 'in', 'this', 'building', '.'], ['all', 'right', ',', 'where', 'is', 'he', '?']), (...)]

    log.info("Counting freq of words...")
    word_counts = collections.Counter()
    for dial in dialogues: # dial: [["he's", 'not', 'in', 'this', 'building', '.'], ['all', 'right', ',', 'where', 'is', 'he', '?']]
        for p in dial: # p: ["he's", 'not', 'in', 'this', 'building', '.']
            word_counts.update(p)

    # Lọc ra lấy những từ có freq xuất hiện nhiều hơn
    # {'apple': 5, 'banana': 3, 'cherry': 7, 'orange': 2} feq_min = 4
    # out: {'apple', 'cherry'}        
    freq_set = set(map(lambda p: p[0], filter(lambda p: p[1] >= min_token_freq, word_counts.items()))) # {'apple', 'cherry'}
    log.info("Data has %d uniq words, %d of them occur more than %d",
             len(word_counts), len(freq_set), min_token_freq)
    
    phrase_dict = phrase_pairs_dict(phrase_pairs, freq_set)
    return phrase_pairs, phrase_dict


def phrase_pairs_dict(phrase_pairs, freq_set): #
    """
    Hàm này giống bag of word nó sẽ đi gắn id cho từng từ vào dict
    Trả về túi từ vựng {"#UK": 0, "#BEG": 1, ...}
    """

    res = {UNKNOWN_TOKEN: 0, BEGIN_TOKEN: 1, END_TOKEN: 2}
    next_id = 3
    for p1, p2 in phrase_pairs: 
    # p1: ["he's", 'not', 'in', 'this', 'building', '.']
    # p2: ['all', 'right', ',', 'where', 'is', 'he', '?']
        for w in map(str.lower, itertools.chain(p1, p2)): #  itertools.chain(p1, p2) ghép 2 chuỗi lại thành một
            if w not in res and w in freq_set:
                res[w] = next_id
                next_id += 1
    return res


def save_emb_dict(dir_name, emb_dict):
    with open(os.path.join(dir_name, EMB_DICT_NAME), "wb") as fd:
        pickle.dump(emb_dict, fd)

def load_emb_dict(dir_name):
    with open(os.path.join(dir_name, EMB_DICT_NAME), "rb") as fd:
        return pickle.load(fd)
    
def encode_words(words, emb_dict):
    res = [emb_dict[BEGIN_TOKEN]]
    unk_idx = emb_dict[UNKNOWN_TOKEN]
    for w in words:
        idx = emb_dict.get(w.lower(), unk_idx) #Từ nào không có sẻ đưa về unk
        res.append(idx)
    res.append(emb_dict[END_TOKEN])
    return res


def encode_phrase_pairs(phrase_pairs, emb_dict, filter_unknows=True):
    '''
    Hàm này sẽ encode các câu phrase_pairs tương ứng emb_dict
    Nó sẽ loại bỏ nhửng câu nào chưa unk trong cặp
    '''

    unk_token = emb_dict[UNKNOWN_TOKEN]
    result = []
    for p1, p2 in phrase_pairs:
        p = encode_words(p1, emb_dict), encode_words(p2, emb_dict)
        if unk_token in p[0] or unk_token in p[1]:
            continue
        result.append(p)
    return result


def group_train_data(training_data):
    """
    Chưa xác định được input
    
    """
    groups = collections.defaultdict(list)
    for p1, p2 in training_data:
        l = groups[tuple(p1)]
        l.append(p2)
    return list(groups.items())


def iterate_batches(data, batch_size):
    """
    Đưa data vào và chia thành các batch
    """

    # Kiểm tra có phải list không
    assert isinstance(data, list)
    # Kiểm tra có phải kiểu int không
    assert isinstance(batch_size, int)

    ofs = 0
    while True:
        batch = data[ofs*batch_size:(ofs+1)*batch_size]

        # Nếu len batch  <= 1 sẽ khong trả về
        if len(batch) <= 1:
            break
        yield batch
        ofs += 1


def decode_words(indices, rev_emb_dict):
    return [rev_emb_dict.get(idx, UNKNOWN_TOKEN) for idx in indices]


def trim_tokens_seq(tokens, end_token):
    res = []
    for t in tokens:
        res.append(t)
        if t == end_token:
            break
    return res


def split_train_test(data, train_ratio=0.95):
    count = int(len(data) * train_ratio)
    return data[:count], data[count:]