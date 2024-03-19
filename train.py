import torch
from models.model.transformer import Transformer
from utils.data_loader import Load_Data, Lang
from config import *
from utils.utils import count_parameters
from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/nlp_ex1')


QA_data = Load_Data(csv_path=csv_path, max_len=MAX_LENGTH, device = device)


obj_lang, train_dataloader = QA_data.get_dataloader(batch_size = batch_size)

model = Transformer(input_size = obj_lang.n_words, hidden_size=hidden_size, vocab_size= obj_lang.n_words, device = device)

count_parameters(model)

x = torch.randint(low=0, high=100, size=(1, 10), dtype=torch.int).to(device)

writer.add_graph(model, x)
writer.close()