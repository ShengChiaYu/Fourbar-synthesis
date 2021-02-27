import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.model_zoo import load_url as load_state_dict_from_url

from utils import parse_args

# model used for v1 and v2
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        ## encoder layers ##
        self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)
        self.pool_1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(16, 8, 3, stride=2, padding=1)
        self.pool_2 = nn.MaxPool2d(2, stride=2)

        ## decoder ##
        self.t_conv1 = nn.ConvTranspose2d(8, 16, 3, stride=3, padding=1)
        self.t_conv2 = nn.ConvTranspose2d(16, 8, 4, stride=3, padding=1)
        self.t_conv3 = nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1)


    def forward(self, x, feature=False):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        # input: (b, 1, 200, 200)
        x = F.relu(self.conv1(x)) # b, 16, 100, 100
        x = self.pool_1(x) # b, 16, 50, 50
        x = F.relu(self.conv2(x)) # b, 8, 25, 25
        # compressed representation
        x = self.pool_2(x) # b, 8, 12, 12
        if feature:
            return x.reshape(x.shape[0],-1)

        ## decode ##
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x)) # b, 16, 34, 34
        x = F.relu(self.t_conv2(x)) # b, 8, 101, 101
        x = torch.sigmoid(self.t_conv3(x)) # b, 1, 200, 200

        return x


def test():
    args = parse_args()

    # model = resnet152(pretrained=True).cuda()
    model = autoencoder().cuda()

    input = torch.rand(4,1,100,100).cuda()
    # input = torch.rand(4,1,28,28).cuda()
    output = model(input)
    print('Tensor size and type:', output.size(), output.dtype)
    # print(model)


if __name__ == '__main__':
    test()
