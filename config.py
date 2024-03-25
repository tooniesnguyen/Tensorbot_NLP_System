import torch
import sys
import socket
import os
from pathlib import Path

device = torch.device("cuda:0")

HOST = socket.gethostbyname(socket.gethostname())
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
WORK_DIR = os.path.dirname(ROOT)

csv_path = f"{WORK_DIR}/Tensorbot_NLP/data/AI.csv"
json_path_train = f"{WORK_DIR}/Tensorbot_NLP/data/train-v2.0.json"
json_path_dev = f"{WORK_DIR}/Tensorbot_NLP/data/dev-v2.0.json"
dict_path = f"{WORK_DIR}/Tensorbot_NLP/data/dicts/lang_obj.pkl"
PATH_SAVE = f"{WORK_DIR}/Tensorbot_NLP/runs/qatask"

path = "runs"
SOS_token = 0
EOS_token = 1


# model parameter setup
MAX_LENGTH = 10
batch_size = 32
hidden_size = 128

if __name__ == "__main__":
    print(WORK_DIR)