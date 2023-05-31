import pandas as pd
import torch.utils.data as data
from sklearn.model_selection import train_test_split

def split_train_val(path):
    data = pd.read_csv(path, header=None)
    train_data, val_data = train_test_split(data, test_size=0.3, random_state=42)
    train_data.to_csv('../../data/interim/train.csv', index=False)
    val_data.to_csv('../../data/processed/val.csv', index=False)
    return train_data, val_data
