import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.utils import resample
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from torchvision import transforms
import pytorch_lightning as pl
from data.dataloader import heartBeatDataset

class heartBeatClass(pl.LightningModule):
	def __init__(self, model):
		super().__init__()
		self.model = model

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
		return optimizer

	def training_step(self, train_batch, batch_idx):
		x, y = train_batch
		loss = F.mse_loss(x_hat, x)
		self.log('train_loss', loss)
		return loss

	def validation_step(self, val_batch, batch_idx):
		x, y = val_batch
	 
		loss = F.mse_loss(x_hat, x)
		self.log('val_loss', loss)

# data
if __name__ == '__main__':
    train_path = '../../data/processed/train_upsampled.csv'
    val_path = '../../data/processed/val.csv'

    try:
        train_set = heartBeatDataset(train_path)
        val_set = heartBeatDataset(val_path, train=False)
    except:
        print('Complete data preprocessing first.')
        exit()
    
    train_loader = DataLoader(train_set, batch_size=32)
    val_loader = DataLoader(val_set, batch_size=32)

    # model
    model = heartBeatClass()
    # training
    trainer = pl.Trainer(accelerator='auto')
    trainer.fit(model, train_loader, val_loader)
    
