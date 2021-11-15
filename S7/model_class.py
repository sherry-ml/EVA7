import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, dropout, normalization='BN'):
        super(Net, self).__init__()
        
        ## Convolution Block1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias = False),  # 32 
            self.norm(normalization,32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),

            nn.Conv2d(32, 64, 4, padding=1, bias = False), # 31 
            self.norm(normalization,64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )
        
        ## Transition Block1
        self.trans1 = nn.Sequential(
            nn.Conv2d(64, 32,3, stride=2), # 15 
            self.norm(normalization,32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

        ## Convolution Block2
        self.conv2 =  nn.Sequential(
            nn.Conv2d(32, 32, 3,  padding=1, bias = False), # 15 
            self.norm(normalization,32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),

            ## Depthwise Seperable Convolution
            nn.Conv2d(32,32, 3,  padding=1,groups=32 ,bias = False),  # 15
            self.norm(normalization,32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )
        
        #Transition Block2
        self.trans2 = nn.Sequential(

            nn.Conv2d(32, 16, 3, stride=2), # 7 
            self.norm(normalization,16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

        #Convolution Block3
        self.conv3 = nn.Sequential(
            
            ## Dilation Block
            nn.Conv2d(16, 32, 3,  padding=1, bias = False,dilation=2), # 5 
            self.norm(normalization,32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),

            nn.Conv2d(32, 32, 3,  padding=1, bias = False),  # 5
            self.norm(normalization,32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

        #Transition Block3
        self.trans3 = nn.Sequential(

            nn.Conv2d(32, 16, 3, stride=2), # 2
            self.norm(normalization,16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
            )

        #Convolution Block4        
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, bias = False), # 2
            self.norm(normalization,16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),

            nn.Conv2d(16, 10, 3, padding=1, bias = False),          # 2
            self.norm(normalization,10),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

        ## Output Block
        self.out = nn.Sequential(
            nn.AvgPool2d(kernel_size=2)
        ) 

    def norm(self,norm_type, channels):
      if(norm_type == 'BN'):
        norm1 = nn.BatchNorm2d(channels)
      elif(norm_type == 'LN'):
        norm1 = nn.GroupNorm(1, channels)
      else:
        norm1 = nn.GroupNorm(2, channels)
      return norm1


    def forward(self, x):
        x = self.conv1(x)
        x = self.trans1(x)

        x = self.conv2(x) 
        x = self.trans2(x) 

        x = self.conv3(x) 
        x = self.trans3(x)

        x = self.conv4(x)
        x = self.out(x)

        x = x.view(-1,10)
        return F.log_softmax(x,dim=1)