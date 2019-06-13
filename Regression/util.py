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
    parser.add_argument('--svm', dest='svm',
                        help='run svm regression',
                        default=0, type=int)

    args = parser.parse_args()

    return args


def load_data(root, transform=False):
    data = np.genfromtxt(root, delimiter=',')
    if transform:
        scaler = MinMaxScaler(copy=True, feature_range=(-1, 1))
        data = scaler.fit_transform(data)

    return data


def load_data_fourbar(data_param):
    positions = data_param['positions']
    target = data_param['target']

    # label_column specifies the index of the column containing the true label
    x_train = load_data(join(os.getcwd(), '../Pytorch/data', positions, 'x_train.csv'), transform=True)
    y_train = load_data(join(os.getcwd(), '../Pytorch/data', positions, 'y_train_param.csv'), transform=False)
    x_test = load_data(join(os.getcwd(), '../Pytorch/data', positions, 'x_test.csv'), transform=True)
    y_test = load_data(join(os.getcwd(), '../Pytorch/data', positions, 'y_test_param.csv'), transform=False)
    print('Training data size:', x_train.shape)
    print('Testing data size:', x_test.shape)

    labels = {'r1':0, 'r3':1, 'r4':2, 'r5':3, 'th6':4}
    label_train = y_train[:, labels[target]]
    label_test = y_test[:, labels[target]]
    print('Training target:', target)

    return x_train, x_test, label_train, label_test


def predict(target, ypred, label, write=True):
    if write:
        save_path = join(os.getcwd(), 'predictions', '{}_pred.csv'.format(target))
        f = open(save_path, "w")
        count = 0
        threshold = 1e-2
        for i in range(ypred.shape[0]):
            diff_ratio = np.abs(ypred[i] - label[i]) / label[i]
            if diff_ratio <= threshold:
                count += 1
            f.write('{0:.4f}, {1:.4f}, {2:.4f}\n'.format(ypred[i], label[i], diff_ratio))
        acc = count/ypred.shape[0] * 100
        print('{0}_acc: {1:.3f}%'.format(target, acc))
        # f.write('{0}_acc: {1:.3f}%'.format(target, acc))
        f.close()

    else:
        threshold = 1e-2
        diff_ratio = np.abs(ypred - label) / label
        count = diff_ratio[diff_ratio <= threshold]
        acc = count.shape[0] / ypred.shape[0] * 100
        print('{0}_acc: {1:.3f}%'.format(target, acc))

    return acc
