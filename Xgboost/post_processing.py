import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

from os.path import join

from util import load_data


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Fourbar Booster')
    # mode
    parser.add_argument('--combine_files', dest='combine_files',
                        help='combine multiple files',
                        default=0, type=int)
    parser.add_argument('--plot', dest='plot',
                        help='plot figures',
                        default=0, type=int)
    args = parser.parse_args()

    return args


def combine_files():
    r1 = load_data(join(os.getcwd(), 'predictions', 'r1_test_pred.csv'), transform=False)
    r3 = load_data(join(os.getcwd(), 'predictions', 'r3_test_pred.csv'), transform=False)
    r4 = load_data(join(os.getcwd(), 'predictions', 'r4_test_pred.csv'), transform=False)
    r5 = load_data(join(os.getcwd(), 'predictions', 'r5_test_pred.csv'), transform=False)
    th6 = load_data(join(os.getcwd(), 'predictions', 'th6_test_pred.csv'), transform=False)

    save_path = join(os.getcwd(), 'predictions', 'all_test_pred.csv')
    f = open(save_path, "w")
    for i in range(r1.shape[0]):
        f.write('{},{},{},{},{}\n'.format(r1[i,0], r3[i,0], r4[i,0], r5[i,0], th6[i,0]))
    f.close()


if __name__ == '__main__':
    args = parse_args()

    if args.combine_files:
        combine_files()
