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
from pytorch_lightning.loggers import WandbLogger
import argparse
sys.path.append('/Users/sushrutdhaka/ecg_heartbt_categorization/src/data')
from dataloader import heartBeatDataset
sys.path.append('/Users/sushrutdhaka/ecg_heartbt_categorization/models')
from model import Classifier


class heartBeatClass(pl.LightningModule):
    def __init__(self, model, lr=1e-3, batch_size=32, seed=42, betas=(0.9,0.999), num_classes=5):
        super().__init__()
        
        self.model = model

        # Metrics
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.valid_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.valid_confusion_matrix = torchmetrics.ConfusionMatrix(task='multiclass',num_classes=self.num_classes)
        self.test_confusion_matrix = torchmetrics.ConfusionMatrix(task='multiclass',num_classes=self.num_classes)

        # Hyperparameters
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.betas = betas
        self.num_classes = num_classes
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=self.betas)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(-1,1,187)
        pred = self.model(x)

        # Loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, y)

        # Calculate and log metrics
        pred = nn.Softmax(dim=1)(pred)
        pred = torch.argmax(pred, dim=1)
        self.train_acc(pred, y)
        self.log('train_loss', loss, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
        self.log('train_acc', self.train_acc, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
        self.log('hyperparam_lr', self.optimizers().param_groups[0]['lr'], on_epoch=True, sync_dist=True, 
                 prog_bar=True, logger=True)

        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(-1,1,187)
        pred = self.model(x)

        # Loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, y)

        # Calculate and log metrics
        pred = nn.Softmax(dim=1)(pred)
        pred = torch.argmax(pred, dim=1)
        self.valid_acc(pred, y)
        self.log('val_loss', loss, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
        self.log('val_acc', self.valid_acc, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
        #self.log('val_confusion_matrix', self.valid_confusion_matrix(pred, y), on_epoch=True, sync_dist=True, prog_bar=True, logger=True)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        x = x.view(-1,1,187)
        pred = self.model(x)

        # Loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, y)

        # Calculate and log metrics
        pred = nn.Softmax(dim=1)(pred)
        pred = torch.argmax(pred, dim=1)
        self.valid_acc(pred, y)
        self.log('test_loss', loss, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
        self.log('test_acc', self.valid_acc, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
        #self.log('test_confusion_matrix', self.valid_confusion_matrix(pred, y), on_epoch=True, sync_dist=True, prog_bar=True, logger=True)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--train_path', type=str, default='../../data/processed/train_upsampled.csv')
    args.add_argument('--val_path', type=str, default='../../data/processed/val.csv')
    args.add_argument('--batch_size', type=int, default=32)
    args.add_argument('--lr', type=float, default=1e-3)
    args.add_argument('--betas', type=tuple, default=(0.9,0.999))
    args.add_argument('--num_classes', type=int, default=5)
    args.add_argument('--max_epochs', type=int, default=100)
    args.add_argument('--seed', type=int, default=42)
    args.add_argument('ckpts', type=str, default='ckpts/', help='path to save checkpoints')
    args = args.parse_args()

    pl.seed_everything(args.seed, workers=True)

    # data loading
    train_path = args.train_path
    val_path = args.val_path 
    try:
        train_set = heartBeatDataset(train_path)
        val_set = heartBeatDataset(val_path, train=False)
        print('Data loading completed.')
    except:
        print('Complete data preprocessing first.')
        exit()
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    # model
    backbone = Classifier(args.num_classes)
    model = heartBeatClass(backbone)

    # training
    wandb_logger = WandbLogger()
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss', save_top_k=5, mode='min', dirpath=args.ckpts,
                                                       filename='model-{epoch:02d}-{val_loss:.2f}')
    trainer = pl.Trainer(devices='auto', accelerator='cpu', max_epochs=args.max_epochs, logger=wandb_logger, 
            log_every_n_steps=1, deterministic=True, callbacks=[pl.callbacks.early_stopping.EarlyStopping
            (monitor='val_loss', patience=10, mode='min'), checkpoint_callback],default_root_dir=args.ckpts)
    
    trainer.fit(model, train_loader, val_loader)

    
    
