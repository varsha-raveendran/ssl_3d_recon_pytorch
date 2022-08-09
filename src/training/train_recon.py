import torch.nn as nn
import torch

class ReconstructionNet(nn.Module):
    def __init__(self, return_feat = False):
        """
        Reconstruct pointcloud from 2d image
        :param 
        """
        
        super().__init__()
        self.return_feat = return_feat
        self.relu  = nn.ReLU()
        
        #Structure branch
        self.cnn1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride=2, padding=(1,1))
        self.cnn2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride=2, padding=(1,1))
        self.cnn3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride=2, padding=(1,1))
        self.cnn4 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride=2, padding=(1,1))
        
        
        self.linear1 = nn.Linear(in_features= 256*4*4 , out_features=128) 
        self.linear2 = nn.Linear(in_features= 128 , out_features=128)
        self.linear3 = nn.Linear(in_features= 128 , out_features=128)
        self.linear4 = nn.Linear(in_features= 128 , out_features=1024*3) #TODO: make 1024 configurable
        
        #Color branch
        self.color_cnn1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride=2, padding=(1,1))
        self.color_cnn2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride=2, padding=(1,1))
        
        self.color_linear1 = nn.Linear(in_features= 64*16*16 , out_features=128)
        self.color_linear2 = nn.Linear(in_features= 128 , out_features=128)
        self.color_linear3 = nn.Linear(in_features= 128 , out_features=128)
        
        self.color_linear4 = nn.Linear(in_features= 256 , out_features=128)
        self.color_linear5 = nn.Linear(in_features= 128 , out_features=1024*3)
        
        self.sigmoid = nn.Sigmoid()
        

    
    def forward(self, img_ip):
        """
        :param x: (3,W,H) tensor #TODO: check
        :return: 1024*3 x 1 tensor
        """
        x = img_ip
        x = self.relu(self.cnn1(x)) #es1
        x = self.relu(self.cnn2(x)) #es2
        x = self.relu(self.cnn3(x)) #es3
        x = self.relu(self.cnn4(x)) #es4
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.relu(self.linear1(x)) #ds1
        # enc_pcl = x
        x = self.relu(self.linear2(x)) #ds2
        x1 = self.relu(self.linear3(x)) #ds3
        x = self.linear4(x1) #ds4
        # print("before view: ", x.shape)
        x = x.view(x.size(0), 3, 1024)        
        # print("after view: ",x.shape)
        
        y = self.relu(self.color_cnn1(img_ip)) #ec1
        y = self.relu(self.color_cnn2(y)) #ec2
        
        y = y.view(y.size(0), -1)
        y = self.relu(self.color_linear1(y)) #dc1
        
        # enc_feat = torch.concat([enc_pcl, y], axis=0)
        # print(enc_feat.shape)
        y = self.relu(self.color_linear2(y)) #dc2
        y = self.relu(self.color_linear3(y))     #dc3   
        # print(x1.shape, " cat " , y.shape)
        y = torch.concat([x1,y], axis=-1) #dc3
          
        y = self.relu(self.color_linear4(y)) #dc4
        y = self.color_linear5(y) #dc4
        
        y = y.view(-1, 1024,3) 
        y = self.sigmoid(y)
        y = y.permute(0, 2,1)
        # print(y.shape , " ", x.shape)    
        #self.z = torch.concat([x,y], axis=1)
        #print("z : " ,self.z.shape)
        # if not self.return_feat:
        return x, y
        # else:
        #     return x, y, enc_feat
        

