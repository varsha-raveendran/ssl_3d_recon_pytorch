import torch.nn as nn
import torch

class ReconstructionNet(nn.Module):
    def __init__(self):
        """
        :param 
        """
        
        super().__init__()
        self.relu  = nn.ReLU()
        
        #Structure branch
        self.cnn1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride=2, padding=(1,1))
        self.cnn2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride=2, padding=(1,1))
        self.cnn3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride=2, padding=(1,1))
        self.cnn4 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride=2, padding=(1,1))
        
        
        self.linear1 = nn.Linear(in_features= 256*14*14 , out_features=128) #TODO: make 1024 configurable
        self.linear2 = nn.Linear(in_features= 128 , out_features=128)
        self.linear3 = nn.Linear(in_features= 128 , out_features=1024*3)
        
    
    def forward(self, x):
        """
        :param x: (3,W,H) tensor #TODO: check
        :return: 1024*3 x 1 tensor
        """
        x = self.relu(self.cnn1(x))
        x = self.relu(self.cnn2(x))
        x = self.relu(self.cnn3(x))
        x = self.relu(self.cnn4(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        x = torch.reshape(x, (-1,))        
        
        return x
        

