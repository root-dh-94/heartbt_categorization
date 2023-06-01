import pandas as pd
import torch.utils.data as data
from sklearn.model_selection import train_test_split
import numpy as np
import torch

class heartBeatDataset(data.Dataset):
    def __init__(self, csv_path, train = True, seed = 42):
        self.data = pd.read_csv(csv_path)
        self.train = train
        self.seed = seed

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        value = np.asarray(self.data.iloc[index, :-1])
        label = self.data.iloc[index, -1]

        if self.train:
            value += np.random.normal(0, 0.1, value.shape)
        
        tensor_value = torch.from_numpy(value)
        tensor_value = tensor_value.float()
        return tensor_value, label
    