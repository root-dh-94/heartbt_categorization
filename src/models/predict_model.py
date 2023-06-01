import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
from dvclive.lightning import DVCLiveLogger
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.utils import resample
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from torchvision import transforms
import pytorch_lightning as pl
import sys
import torchmetrics
from train_model import heartBeatClass
from pytorch_lightning.loggers import WandbLogger
sys.path.append('/Users/sushrutdhaka/ecg_heartbt_categorization/src/data')
from dataloader import heartBeatDataset
sys.path.append('/Users/sushrutdhaka/ecg_heartbt_categorization/models')
from model import Classifier


if __name__ == '__main__':
    pl.seed_everything(42, workers=True)

    # Load data
    test_path = '/Users/sushrutdhaka/ecg_heartbt_categorization/data/processed/test.csv'

    try:
        test_set = heartBeatDataset(test_path)
        print('Data loading completed.')
    except:
        print('Complete data preprocessing first.')
        exit()

    test_loader = DataLoader(test_set, batch_size=32)

    # model
    backbone = Classifier(5)
    model = heartBeatClass(backbone)
    # training
    wandb_logger = WandbLogger()
    
    trainer = pl.Trainer(devices='auto', accelerator='cpu', max_epochs=100, logger=wandb_logger, 
            log_every_n_steps=1, deterministic=True)
    trainer.test(model, test_loader)
