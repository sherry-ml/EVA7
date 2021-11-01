import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, normalization='BN'):

      super().__init__()
     
      self.conv1 = nn.Sequential(nn.Conv2d(1, 8, 3, bias=False),
                                 self.norm(normalization,8),
                                 nn.ReLU(inplace=True), #8*26*26
                                 nn.Dropout(0.02),
                                 nn.Conv2d(8, 16, 3, bias=False),
                                 self.norm(normalization,16),
                                 nn.ReLU(inplace=True), #16*24*24
                                 nn.Dropout(0.02)
      )

      self.trans = nn.Sequential(nn.MaxPool2d(2,2), #16*12*12
                                 nn.Conv2d(16,8,1, bias=False),
                                 self.norm(normalization,8),
                                 nn.ReLU(inplace=True), #8*12*12
                                 nn.Dropout(0.02)
                                 )

      self.conv2 = nn.Sequential(nn.Conv2d(8, 8, 3, bias=False),
                                 self.norm(normalization,8),
                                 nn.ReLU(inplace=True), #8*10*10
                                 nn.Dropout(0.02),
                                 nn.Conv2d(8, 16, 3, bias=False),
                                 self.norm(normalization,16),
                                 nn.ReLU(inplace=True), #16*8*8
                                 nn.Dropout(0.02),
                                 nn.Conv2d(16, 16, 3, bias=False),#16*6*6
                                 self.norm(normalization,16),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(0.02)
                                 )

      self.output = nn.Sequential(nn.Conv2d(16, 10, 1, bias=False)) #10*6*6
      self.avgpool2d = nn.AvgPool2d(kernel_size=6) #10*1*1

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
        x = self.trans(x)
        x = self.conv2(x)
        x = self.output(x)
        x = self.avgpool2d(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)