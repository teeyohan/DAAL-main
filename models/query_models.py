
'''
Learning loss for active learning.
Yoo, Donggeun, In So Kweon.
CVPR, 2019.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import * 


class LossNet(nn.Module):
    def __init__(self, feature_sizes=[32, 64, 128, 256], num_channels=[64, 128, 256, 512], interm_dim=128):
        super(LossNet, self).__init__()
        self.conv1 = nn.Conv2d(256, 64, kernel_size=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=1, stride=1, bias=False)
        self.conv3 = nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False)
        self.conv4 = nn.Conv2d(32, 512, kernel_size=1, stride=1, bias=False)

        self.GAP1 = nn.AvgPool2d(feature_sizes[0])
        self.GAP2 = nn.AvgPool2d(feature_sizes[1])
        self.GAP3 = nn.AvgPool2d(feature_sizes[2])
        self.GAP4 = nn.AvgPool2d(feature_sizes[3])

        self.FC1 = nn.Linear(num_channels[0], interm_dim)
        self.FC2 = nn.Linear(num_channels[1], interm_dim)
        self.FC3 = nn.Linear(num_channels[2], interm_dim)
        self.FC4 = nn.Linear(num_channels[3], interm_dim)

        self.linear = nn.Linear(4 * interm_dim, 1)

    def forward(self, features): 
        out1 = self.conv1(features[0]) 
        out1 = self.GAP1(out1) 
        out1 = out1.view(out1.size(0), -1) 
        out1 = F.relu(self.FC1(out1))

        out2 = self.conv2(features[1]) 
        out2 = self.GAP2(out2)
        out2 = out2.view(out2.size(0), -1)
        out2 = F.relu(self.FC2(out2))

        out3 = self.conv3(features[2])
        out3 = self.GAP3(out3)
        out3 = out3.view(out3.size(0), -1)
        out3 = F.relu(self.FC3(out3))

        out4 = self.conv4(features[3])
        out4 = self.GAP4(out4)
        out4 = out4.view(out4.size(0), -1)
        out4 = F.relu(self.FC4(out4))

        out = self.linear(torch.cat((out1, out2, out3, out4), 1))

        return out 

# ########################################


class TDNet(nn.Module):
    def __init__(self, feature_sizes=[32, 64, 128, 256], num_channels=[64, 128, 256, 512], interm_dim=128, class_num=2):
        super(TDNet, self).__init__()

        self.conv1 = nn.Conv2d(256, 64, kernel_size=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=1, stride=1, bias=False)
        self.conv3 = nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False)
        self.conv4 = nn.Conv2d(32, 512, kernel_size=1, stride=1, bias=False)

        self.GAP1 = nn.AvgPool2d(feature_sizes[0])
        self.GAP2 = nn.AvgPool2d(feature_sizes[1])
        self.GAP3 = nn.AvgPool2d(feature_sizes[2])
        self.GAP4 = nn.AvgPool2d(feature_sizes[3])

        self.FC1 = nn.Linear(num_channels[0], interm_dim)
        self.FC2 = nn.Linear(num_channels[1], interm_dim)
        self.FC3 = nn.Linear(num_channels[2], interm_dim)
        self.FC4 = nn.Linear(num_channels[3], interm_dim)

        self.linear = nn.Linear(4 * interm_dim, 2*256*256)

    def forward(self,  features):
        
        out1 = self.conv1(features[0])
        out1 = self.GAP1(out1)
        out1 = out1.view(out1.size(0), -1)
        out1 = F.relu(self.FC1(out1))

        out2 = self.conv2(features[1])
        out2 = self.GAP2(out2)
        out2 = out2.view(out2.size(0), -1)
        out2 = F.relu(self.FC2(out2))

        out3 = self.conv3(features[2])
        out3 = self.GAP3(out3)
        out3 = out3.view(out3.size(0), -1)
        out3 = F.relu(self.FC3(out3))

        out4 = self.conv4(features[3])
        out4 = self.GAP4(out4)
        out4 = out4.view(out4.size(0), -1)
        out4 = F.relu(self.FC4(out4))
       
        out = self.linear(torch.cat((out1, out2, out3, out4), 1))
        out = out.view(out.size(0), 2, 256, 256)

        return out

