import torch
import sys
import socket
import os
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HOST = socket.gethostbyname(socket.gethostname())
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
WORK_DIR = os.path.dirname(ROOT)

csv_path = f"{WORK_DIR}/Tensorbot_NLP/data/AI.csv"
json_path_train = f"{WORK_DIR}/Tensorbot_NLP/data/train-v2.0.json"
json_path_dev = f"{WORK_DIR}/Tensorbot_NLP/data/dev-v2.0.json"
dict_path_json = f"{WORK_DIR}/Tensorbot_NLP/data/dicts/lang_obj_json.pkl"
dict_path_csv = f"{WORK_DIR}/Tensorbot_NLP/data/dicts/lang_obj_csv.pkl"
PATH_SAVE = f"{WORK_DIR}/Tensorbot_NLP/runs/qatask"
PATH_SAVE_CSV = f"{WORK_DIR}/Tensorbot_NLP/runs/qatask_csv"
PATH_SAVE_RL = f"{WORK_DIR}/Tensorbot_NLP/runs/qatask_rl_5lr"
PATH_SAVE_RL_CSV = f"{WORK_DIR}/Tensorbot_NLP/runs/qatask_rlcsv"

path = "runs"
SOS_token = 0
EOS_token = 1


# model parameter setup
MAX_LENGTH = 10
batch_size = 1024
hidden_size = 128


learning_rate = 5e-4
epoch_th = 100
num_samples = 5


if __name__ == "__main__":
    print(WORK_DIR)