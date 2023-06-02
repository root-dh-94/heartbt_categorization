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
import argparse
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


if __name__ == '__main__':
    print('Testing model...')
    args = argparse.ArgumentParser()
    args.add_argument('--test_path', type=str, default='../../data/processed/test.csv')
    args.add_argument('--num_classes', type=int, default=5)
    args.add_argument('--seed', type=int, default=42)
    args.add_argument('--batch_size', type=int, default=32)
    args.add_argument('--checkpoint_file', type=str, help='path to load checkpoints')
    args = args.parse_args()
    pl.seed_everything(42, workers=True)

    # Load data
    test_path = args.test_path

    try:
        test_set = heartBeatDataset(test_path)
        print('Data loading completed.')
    except:
        print('Complete data preprocessing first.')
        exit()

    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    # model
    backbone = Classifier(args.num_classes)
    model = heartBeatClass(backbone)
    # training
    wandb_logger = WandbLogger()
    
    trainer = pl.Trainer(devices='auto', accelerator='gpu',logger=wandb_logger, 
            log_every_n_steps=1, deterministic=True, )
    model_output = trainer.predict(model, test_loader,
                  ckpt_path=args.checkpoint_file)
    
    # unpack model output to measure performance metrics
    prediction_list = []
    label_list = []
    for batch in model_output:
        predictions, labels = batch
        for pred in predictions:
            prediction_list.append(pred)
        for label in labels:
            label_list.append(label)

    # print performance metrics
    print('Classification Report: ')
    print(classification_report(label_list, prediction_list))
    print('Accuracy of the model on the Test Dataset is: ', accuracy_score(label_list, prediction_list))
    print('Confusion Matrix: ')
    print(confusion_matrix(label_list, prediction_list))

    #plot normalized confusion matrix
    cm = confusion_matrix(label_list, prediction_list)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
 


   
    
