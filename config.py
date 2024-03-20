import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



csv_path = "/home/toonies/Learn/Tensorbot_NLP/data/AI.csv"
path = "runs"
SOS_token = 0
EOS_token = 1


# model parameter setup
MAX_LENGTH = 10
batch_size = 32
hidden_size = 128