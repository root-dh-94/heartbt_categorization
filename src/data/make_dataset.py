import pandas as pd
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

def preprocess_data(path, train, seed=42):
    #prepare train and val data, and relname label column
    data = pd.read_csv(path, header=None)
    data.rename(columns = {187: 'labels'}, inplace = True)
    data.labels = data.labels.astype(int)

    if train:
        train_data, val_data = train_test_split(data, test_size=0.3, random_state=42)
        train_data.to_csv('../../data/interim/train.csv', index=False)
        val_data.to_csv('../../data/processed/val.csv', index=False)

    else:
        data.to_csv('../../data/processed/test.csv', index=False)

    print('Data preprocessing completed.')
    return


def train_resample(train_data_path, samples_class=10000, seed=42):
    #upsample minority class to balance training dataset
    train_data = pd.read_csv(train_data_path)
    train_data_majority = train_data[train_data.labels==0]
    train_data_minority_S = train_data[train_data.labels==1]
    train_data_minority_V = train_data[train_data.labels==2]
    train_data_minority_F = train_data[train_data.labels==3]
    train_data_minority_Q = train_data[train_data.labels==4]

    train_data_majority = train_data_majority.sample(n=samples_class, random_state=seed)
    train_data_minority_upsampled_S = resample(train_data_minority_S, replace=True, n_samples=samples_class, random_state=seed)
    train_data_minority_upsampled_V = resample(train_data_minority_V, replace=True, n_samples=samples_class, random_state=seed)
    train_data_minority_upsampled_F = resample(train_data_minority_F, replace=True, n_samples=samples_class, random_state=seed)
    train_data_minority_upsampled_Q = resample(train_data_minority_Q, replace=True, n_samples=samples_class, random_state=seed)

    train_data_upsampled = pd.concat([train_data_majority, train_data_minority_upsampled_S, 
                                      train_data_minority_upsampled_V, train_data_minority_upsampled_F, 
                                      train_data_minority_upsampled_Q])
    
    train_data_upsampled.to_csv('../../data/processed/train_upsampled.csv', index=False)
    print('Training data resampling completed.')
    return

if __name__ == '__main__':
    preprocess_data('../../data/raw/heartbt_data/mitbih_train.csv', train=True)
    preprocess_data('../../data/raw/heartbt_data/mitbih_test.csv', train=False)
    train_resample('../../data/interim/train.csv')