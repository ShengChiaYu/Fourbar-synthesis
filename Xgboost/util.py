import os
import sys
import numpy as np
import argparse

from os.path import join
from sklearn.preprocessing import MinMaxScaler

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Fourbar Booster')
    # Load data
    parser.add_argument('--pos_num', dest='pos_num',
                        help='60, 360',
                        default='60positions', type=str)
    parser.add_argument('--target', dest='tar',
                        help='r1, r3, r4, r5, th6',
                        default='r1', type=str)

    # Training setup
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=1000, type=int)
    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models',
                        default="models", type=str)
    parser.add_argument('--early_stopping_rounds', dest='early_stopping_rounds',
                        help='early_stopping_rounds',
                        default=10, type=int)

    # Configure optimization
    parser.add_argument('--max_depth', dest='max_depth',
                        help='max_depth of tree',
                        default=6, type=int)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.3, type=float)

    # mode
    parser.add_argument('--load_model', dest='load_model',
                        help='load saved model',
                        default=0, type=int)
    parser.add_argument('--tune', dest='tune',
                        help='tuning mode',
                        default=0, type=int)

    args = parser.parse_args()

    return args


def load_data(root, transform=False):
    data = np.genfromtxt(root, delimiter=',')
    if transform:
        scaler = MinMaxScaler(copy=True, feature_range=(-1, 1))
        data = scaler.fit_transform(data)

    return data
