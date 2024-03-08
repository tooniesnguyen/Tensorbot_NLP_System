import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


SOS_token = 0
EOS_token = 1

MAX_LENGTH = 10
