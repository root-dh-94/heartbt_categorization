import torch
from torch import nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

    def forward(self, x):
        return output