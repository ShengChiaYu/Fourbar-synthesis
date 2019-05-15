import os
import sys
import glob
import os.path

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class FDs(Dataset):
    max = 1
    min = -1
    def __init__(self, root, x_file, y_file, transform):
        """ Intialize the Fourier descriptors dataset """
        self.root = root
        self.x_file = x_file
        self.y_file = y_file
        self.x_data = []
        self.y_data = []
        self.x_scaler = None
        self.y_scaler = None
        self.x_std = None
        self.y_std = None
        self.transform = transform

        # read files to get scalers
        fx = open(os.path.join(self.root, self.x_file), 'r')
        fy = open(os.path.join(self.root, self.y_file), 'r')
        lines = fx.readlines()
        for line in lines:
            tmp = [float(i) for i in line.split(',')]
            self.x_data.append(tmp)
        lines = fy.readlines()
        for line in lines:
            tmp = [float(i) for i in line.split(',')]
            self.y_data.append(tmp)
        self.x_data = torch.Tensor(self.x_data)
        self.y_data = torch.Tensor(self.y_data)

        if self.transform:
            self.x_scaler = (torch.max(self.x_data,0)[0], torch.min(self.x_data,0)[0])
            # self.y_scaler = (torch.max(self.y_data,0)[0], torch.min(self.y_data,0)[0])

            self.x_std = (self.x_data - self.x_scaler[1]) / (self.x_scaler[0] - self.x_scaler[1])
            # self.y_std = (self.y_data - self.y_scaler[1]) / (self.y_scaler[0] - self.y_scaler[1])

            self.x_data = self.x_std * (self.max - self.min) + self.min
            # self.y_data = self.y_std * (self.max - self.min) + self.min

        self.len = self.x_data.size(0)

    def __getitem__(self, index):
        """ Get a sample from the dataset """

        return self.x_data[index], self.y_data[index]

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len

def test():
    root = os.path.join('data','60positions')
    y_file = '_param.csv'
    print('root:', root, ', y file:', y_file)

    testset = FDs(root=root, x_file='x_test.csv', y_file='y_test'+y_file, transform=True)
    inputs, targets = testset.__getitem__(9098) # max(9098,1) min(9355,1)
    print(inputs)
    print(targets)

if __name__ == '__main__':
    test()
