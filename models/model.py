import torch
from torch import nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, output_size, hidden_size = 64):
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size

        # Convolutional layers
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1))
        self.pre_residual_layer_1 = nn.Sequential(nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, 
                                                          stride=1, padding = 2), nn.BatchNorm1d(32), nn.ReLU(), 
                                                          nn.Dropout(0.2), nn.Conv1d(in_channels=32, 
                                                          out_channels=32, kernel_size=5, stride=1, padding = 2))
        
        # Residual Blocks
        self.post_residual_layer_1 = nn.Sequential(nn.ReLU(), nn.MaxPool1d(kernel_size=5, stride=2))

        self.pre_residual_layer_2 = nn.Sequential(nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, 
                                                          stride=1, padding = 2), nn.BatchNorm1d(32), nn.ReLU(), 
                                                          nn.Dropout(0.2), nn.Conv1d(in_channels=32,
                                                          out_channels=32, kernel_size=5, stride=1, padding = 2))
        
        self.post_residual_layer_2 = nn.Sequential(nn.ReLU(), nn.MaxPool1d(kernel_size=5, stride=2))

        self.pre_residual_layer_3 = nn.Sequential(nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, 
                                                          stride=1, padding = 2), nn.BatchNorm1d(32), nn.ReLU(), 
                                                          nn.Dropout(0.2), nn.Conv1d(in_channels=32, 
                                                          out_channels=32, kernel_size=5, stride=1, padding = 2))
        
        self.post_residual_layer_3 = nn.Sequential(nn.ReLU(), nn.MaxPool1d(kernel_size=5, stride=2))

        self.pre_residual_layer_4 = nn.Sequential(nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, 
                                                          stride=1, padding = 2), nn.BatchNorm1d(32), nn.ReLU(), 
                                                          nn.Dropout(0.2), nn.Conv1d(in_channels=32,
                                                          out_channels=32, kernel_size=5, stride=1, padding = 2))
        
        self.post_residual_layer_4 = nn.Sequential(nn.ReLU(), nn.MaxPool1d(kernel_size=5, stride=2))
        
        self.pre_residual_layer_5 = nn.Sequential(nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, 
                                                          stride=1, padding = 2), nn.BatchNorm1d(32), nn.ReLU(), 
                                                          nn.Dropout(0.2), nn.Conv1d(in_channels=32,
                                                          out_channels=32, kernel_size=5, stride=1, padding = 2))
        
        self.post_residual_layer_5 = nn.Sequential(nn.ReLU(), nn.MaxPool1d(kernel_size=5, stride=2))

        # Fully connected layers
        self.fc1 = nn.Sequential(nn.Dropout(0.5), nn.Linear(self.hidden_size, 32), nn.ReLU(), nn.Dropout(0.5))
        self.fc2 = nn.Linear(32, self.output_size)

    def forward(self, x):
        # Input Convolutional layer
        conv_layer = self.conv1(x)

        # Residual Blocks
        residual_block1_part1 = self.pre_residual_layer_1(conv_layer)
        residual_block1_part2 = self.post_residual_layer_1(conv_layer + residual_block1_part1)

        residual_block2_part1 = self.pre_residual_layer_2(residual_block1_part2)
        residual_block2_part2 = self.post_residual_layer_2(residual_block1_part2 + residual_block2_part1)

        residual_block3_part1 = self.pre_residual_layer_3(residual_block2_part2)
        residual_block3_part2 = self.post_residual_layer_3(residual_block2_part2 + residual_block3_part1)

        residual_block4_part1 = self.pre_residual_layer_4(residual_block3_part2)
        residual_block4_part2 = self.post_residual_layer_4(residual_block3_part2 + residual_block4_part1)

        residual_block5_part1 = self.pre_residual_layer_5(residual_block4_part2)
        residual_block5_part2 = self.post_residual_layer_5(residual_block4_part2 + residual_block5_part1)

        # Flatten otput for fully connected layers
        fc1_input = nn.Flatten()(residual_block5_part2)
        fc1_output = self.fc1(fc1_input)
        output = self.fc2(fc1_output)
        return output
        
