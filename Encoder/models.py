import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.model_zoo import load_url as load_state_dict_from_url

from utils import parse_args


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        ## encoder layers ##
        self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=0)
        self.pool_1 = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.conv2 = nn.Conv2d(16, 8, 3, stride=2, padding=0)
        self.pool_2 = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.conv3 = nn.Conv2d(8, 4, 3, stride=2, padding=0)
        self.pool_3 = nn.MaxPool2d(2, stride=2, return_indices=True)

        ## decoder ##
        self.unpool_1 = nn.MaxUnpool2d(3, stride=3, padding=0)
        self.t_conv1 = nn.ConvTranspose2d(4, 8, 4, stride=2, padding=1)
        self.unpool_2 = nn.MaxUnpool2d(2, stride=2, padding=0)
        self.t_conv2 = nn.ConvTranspose2d(8, 16, 3, stride=2, padding=0)
        self.unpool_3 = nn.MaxUnpool2d(3, stride=2, padding=0)
        self.t_conv3 = nn.ConvTranspose2d(16, 1, 4, stride=2, padding=0)




    def forward(self, x, feature=False):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        # input: (b, 1, 200, 200)
        x = F.relu(self.conv1(x)) # b, 16, 99, 99
        # print(x.shape)
        x, indices_1 = self.pool_1(x) # b, 16, 49, 49
        # print(x.shape)
        x = F.relu(self.conv2(x)) # b, 8, 24, 24
        # print(x.shape)
        x, indices_2 = self.pool_2(x) # b, 8, 12, 12
        # print(x.shape)
        x = F.relu(self.conv3(x)) # b, 4, 5, 5
        # print(x.shape)
        x, indices_3 = self.pool_3(x) # b, 4, 2, 2 compressed representation
        # print(x.shape)

        if feature:
            return x.reshape(x.shape[0],-1)

        ## decode ##
        # add transpose conv layers, with relu activation function
        x = self.unpool_1(x, indices_3) # b, 4, 6, 6
        # print(x.shape)
        x = F.relu(self.t_conv1(x)) # b, 8, 12, 12
        # print(x.shape)
        x = self.unpool_2(x, indices_2) # b, 8, 24, 24
        # print(x.shape)
        x = F.relu(self.t_conv2(x)) # b, 16, 49, 49
        # print(x.shape)
        x = self.unpool_3(x, indices_1) # b, 16, 99, 99
        # print(x.shape)
        x = torch.sigmoid(self.t_conv3(x)) # b, 1, 200, 200
        # print(x.shape)

        return x


def test():
    args = parse_args()

    # model = resnet152(pretrained=True).cuda()
    model = autoencoder().cuda()

    input = torch.rand(4,1,200,200).cuda()
    # input = torch.rand(4,1,28,28).cuda()
    output = model(input)
    # print('Tensor size and type:', output.size(), output.dtype)
    # print(model)


if __name__ == '__main__':
    test()
