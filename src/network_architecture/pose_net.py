import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class PoseNet(nn.Module):
    """Predict pose from the given rendered image"""
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            #224x224
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=(1,1)),
            nn.ReLU(),
            #112x112
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=(1,1)),
            nn.ReLU(),
            #56x56
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=(1,1)),
            nn.ReLU(),
            #28x28
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=(1,1)),
            nn.ReLU()
            #14x14
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=4*4*256, out_features=128),
            nn.ReLU(),

            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),

            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),

            nn.Linear(in_features=128, out_features=2),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        x = torch.pi*x
        return x.view(-1,2)

# model = PoseNet().cuda()
# print(summary(model,(3,224,224)))        