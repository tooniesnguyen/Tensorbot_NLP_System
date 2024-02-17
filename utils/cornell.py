import collections
import logging
import os
import pickle
from . import utils

log = logging.getLogger("data")

EMB_DICT_NAME= "emb_dict.dat"

DATA_DIR = "data/cornell"
SEPARATOR =  "+++$+++"


def load_dialogues(data_dir=DATA_DIR, genre_filter=''):
    """
    Load dialogues from cornell data
    :return: list of list of list of words

    """
    movie_set = None
    if genre_filter:
        movie_set = read_movie_set(data_dir, genre_filter) # Trả vè các m_id
        log.info("Loaded %d movies with genre %s", len(movie_set), genre_filter)
    log.info("Read and tokenise phrases...")
    lines = read_phrases(data_dir, movies=movie_set) # {"L0": "We are groot .."}
    log.info("Loaded %d phrases", len(lines))
    dialogues = load_conversations(data_dir, lines, movie_set)
    return dialogues


def load_conversations(data_dir, lines,  movies=None):
    """
    This function will load all couple conversation in dict lines suitable with movies
    
    """
    res = []
    for parts in iterate_entries(data_dir, "movie_conversations.txt"):
        m_id, dial_s = parts[2], parts[3]
        if m_id not in movies:
            continue
        l_ids = dial_s.strip("[]").split(", ") # "['L198', 'L199']" => ["'L511'", "'L512'"]
        l_ids = list(map(lambda s: s.strip("'"), l_ids)) # => ['L926', 'L927']
        dial = [lines[l_id] for l_id in l_ids if l_id in lines] # [["he's", 'not', 'in', 'this', 'building', '.'], ['all', 'right', ',', 'where', 'is', 'he', '?']]
        if dial:
            res.append(dial)
        # print(dial)
    return res


def read_phrases(data_dir, movies=None):
    """
    This function will return dict of m0 inclucde line and string of that line
    Input: "m0"
    Ouput: {L1: "asda as "} 
    
    """

    res = {}
    for parts in iterate_entries(data_dir, "movie_lines.txt"):
        l_id, m_id, l_str = parts[0], parts[2], parts[4]
        if m_id not in movies:
            continue
        tokens = utils.tokenize(l_str)
        if tokens:
            res[l_id] = tokens

    return res


def iterate_entries(data_dir, file_name):
    """
    Input: u0 +++$+++ BIANCA +++$+++ m0 +++$+++ 10 things i hate about you +++$+++ f +++$+++ 4
    Output: ['m0', '10 things i hate about you', '1999', '6.90', '62847', "['comedy', 'romance']"]
    """
    with open(os.path.join(data_dir, file_name), "rb") as fd:
        for l in fd:
            l = str(l, encoding='utf-8', errors='ignore')
            yield list(map(str.strip, l.split(SEPARATOR)))


def read_movie_set(data_dir, genre_filter):
    '''
    Hàm này để  lọc ra các thể loại mình muốn train để giảm thiểu thời gian train
    Return các m_id
    '''
    res = set()
    for parts in iterate_entries(data_dir, "movie_titles_metadata.txt"):
        m_id, m_genres = parts[0], parts[5]
        if m_genres.find(genre_filter) != -1:
            res.add(m_id)
    return res

def read_genres(data_dir):
    """
    Create dict type of m_x
    {"m0": ["comedy], "drama", ....}
    """
    res = {}
    for parts in iterate_entries(data_dir,  "movie_titles_metadata.txt"):
        m_id, m_genres = parts[0], parts[5]
        l_genres = m_genres.strip("[]").split(", ")
        l_genres = list(map(lambda s: s.strip("'"), l_genres))
        res[m_id] = l_genres
    return res


def save_emb_dict(dir_name, emb_dict):
    with open(os.path.join(dir_name, EMB_DICT_NAME), "wb") as fd:
        # lưu dict dưới dạng nhị phân
        pickle.dump(emb_dict, fd)


if __name__ == "__main__":
    load_dialogues(DATA_DIR, "drama")