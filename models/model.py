import torch
from torch import nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_size = 256):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1)
        self.pre_residual_layer = nn.Sequential(nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, 
                                                          stride=1), nn.ReLU(), nn.Conv1d(in_channels=32, 
                                                          out_channels=32, kernel_size=5, stride=1))
        
        self.post_residual_layer = nn.Sequential(nn.ReLU(), nn.MaxPool1d(kernel_size=5, stride=2, padding=2))
        self.fc1 = nn.Sequential(nn.Linear(self.hidden_size, 32), nn.ReLU())
        self.fc2 = nn.Linear(32, self.output_size)


    def forward(self, x):
        conv_layer = self.conv1(x)

        residual_block1_part1 = self.pre_residual_layer(conv_layer)
        residual_block1_part2 = self.post_residual_layer(conv_layer + residual_block1_part1)

        residual_block2_part1 = self.pre_residual_layer(residual_block1_part2)
        residual_block2_part2 = self.post_residual_layer(residual_block1_part2 + residual_block2_part1)

        residual_block3_part1 = self.pre_residual_layer(residual_block2_part2)
        residual_block3_part2 = self.post_residual_layer(residual_block2_part2 + residual_block3_part1)

        residual_block4_part1 = self.pre_residual_layer(residual_block3_part2)
        residual_block4_part2 = self.post_residual_layer(residual_block3_part2 + residual_block4_part1)

        residual_block5_part1 = self.pre_residual_layer(residual_block4_part2)
        residual_block5_part2 = self.post_residual_layer(residual_block4_part2 + residual_block5_part1)

        fc1_input = nn.Flatten()(residual_block5_part2)
        fc1_output = self.fc1(fc1_input)
        output = self.fc2(fc1_output)
        return output
        
