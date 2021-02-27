import os
import sys
import glob

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from joblib import dump, load
from tqdm import tqdm

from preprocessing import efd_fitting, curve_normalization_pca2
from utils import parse_args


class FDs(Dataset):
    def __init__(self, root, scale=[], select_dir='GCRR', train=True, valid=False):
        """ Intialize the Fourier descriptors dataset """
        self.train = train
        if self.train:
            dirs = {'GCCC':[1,2,3,5,6], 'GCRR':[0,2,3,5,6],'GRCR':[0,1,3,5,6],'GRRC':[0,1,2,5,6],
                    'RRR1':[0,2,3,5,6], 'RRR2':[1,2,3,5,6],'RRR3':[2,1,3,5,6],'RRR4':[3,1,2,5,6]}
            slice = dirs[select_dir]

            data = load(root)
            L, p1s, sps = data[0], data[1], data[2]

            if valid:
                self.scale = scale
                sps = self.scale.transform(sps)
            else:
                self.scale = StandardScaler()
                self.scale.fit(sps)
                sps = self.scale.transform(sps)

            self.x_data = torch.from_numpy(sps)
            self.y_data = [torch.from_numpy(L[:,slice]), torch.from_numpy(p1s)]

        else:
            data = load(root)
            fns, p1s, sps = data[0], data[1], data[2]

            self.scale = scale
            sps = self.scale.transform(sps)

            self.fn_data = fns
            self.x_data = torch.from_numpy(sps)
            self.y_data = torch.from_numpy(p1s)

        self.len = self.x_data.shape[0]
        # print(self.len)


    def __getitem__(self, index):
        """ Get a sample from the dataset """
        if self.train:
            return self.x_data[index], self.y_data[0][index], self.y_data[1][index]
        else:
            return self.fn_data[index], self.x_data[index], self.y_data[index]



    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len


class multi_FDs(Dataset):
    def __init__(self, root, train=True):
        """ Intialize the Fourier descriptors dataset """
        inversions = [('GCCC',7),('GCRR',6),('GRCR',9),('GRRC',8),('RRR1',4),('RRR2',4),('RRR3',9),('RRR4',6)]
        self.train = train
        self.x_data = []
        self.y_data = []

        if self.train:
            grashof_data_name = 'data_b_1215_14560_{}.joblib'
            non_grashof_data_names = 'data_b_1215_12896_{}.joblib'
        else:
            grashof_data_name = 'data_b_1215_2496_{}.joblib'
            non_grashof_data_names = 'data_b_1215_2496_{}.joblib'

        for i in range(len(inversions)):
            if i < 4:
                data_path = os.path.join(root, inversions[i][0], grashof_data_name.format(inversions[i][1]))
            else:
                data_path = os.path.join(root, inversions[i][0], non_grashof_data_names.format(inversions[i][1]))

            print(data_path)
            data = load(data_path)
            _, _, sps = data[0], data[1], data[2]
            self.x_data.append(torch.from_numpy(sps))
            self.y_data.append(torch.from_numpy(np.full((sps.shape[0],1),i)))

        self.x_data = torch.cat(self.x_data)
        self.y_data = torch.cat(self.y_data).squeeze()

        self.len = self.x_data.shape[0]


    def __getitem__(self, index):
        """ Get a sample from the dataset """
        return self.x_data[index], self.y_data[index]


    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len


def test():
    args = parse_args()

    data_set = ['data_1215_14560.joblib', 'data_1215_12896.joblib',
                'data_1215_2496.joblib', 'testing_data.joblib']

    mechanism = 'RRR2'
    root = os.path.join('data/{}'.format(mechanism), data_set[1])
    print("Load ", root)
    trainset = FDs(root=root, select_dir=mechanism)
    sps, L, p1s = trainset.__getitem__(0)
    print(L)
    print(p1s.shape)

    root = os.path.join('data/{}'.format(mechanism), data_set[2])
    print("Load ", root)
    valset = FDs(root=root, select_dir=mechanism)
    sps, L, p1s = valset.__getitem__(0)
    print(L)


def test_multi_FDs():
    args = parse_args()
    trainset = multi_FDs(root=args.data_dir, train=True)
    inputs, targets = trainset.__getitem__(0)
    print(inputs[:10])
    print(targets)

    valset = multi_FDs(root=args.data_dir, train=False)
    inputs, targets = valset.__getitem__(0)
    print(inputs[:10])
    print(targets)


if __name__ == '__main__':
    test()
    # test_multi_FDs()
