import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

input_size = 5
output_size = 2

batch_size = 30
data_size = 100

device = torch.device("cuda:0" if torch.cuda.is_avaliable() else "cpu")

class RandomDataset(Datast):
