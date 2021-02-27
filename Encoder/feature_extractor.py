import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import glob
import os
import sys
import argparse
from os.path import join

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import image_data_3
from models import autoencoder
from utils import AverageMeter, parse_args


def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    # optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)


def feature_extractor(model, trainset_loader, args):
    features = open(join(args.save_dir, 'features_{}_{}.txt'.format(args.select_dir, args.net)), 'w')
    model.eval()

    train_pbar = tqdm(total=len(trainset_loader), ncols=10, leave=True)
    for batch_idx, (nums, inputs, _, _) in enumerate(trainset_loader):
        if args.use_cuda:
            inputs = inputs.cuda(args.gpu_id)

        outputs = model(inputs, feature=True)
        for i in range(len(nums)):
            features.write('{},'.format(nums[i]))
            feature = ['{},'.format(val) for val in outputs[i].tolist()]
            features.writelines(feature)
            features.write('\n')

        train_pbar.update()
        features.flush()

    train_pbar.close()
    features.close()


if __name__ == '__main__':
    # parse args
    args = parse_args()

    args.use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(args.gpu_id) if args.use_cuda else "cpu")
    print(device)

    root = join(os.getcwd(), 'image_data_3')
    print('root:', root)

    if args.select_dir is not None:
        dirs = {'GCCC':0, 'GCRR':1,'GRCR':2,'GRRC':3}
        select_dir = dirs[args.select_dir]
    else:
        select_dir = None

    transform=transforms.Compose([
              # transforms.RandomCrop(args.img_size, padding=None, pad_if_needed=True, fill=0, padding_mode='edge'),
              transforms.ToTensor(),
    ])
    trainset = image_data_3(root=root, transform=transform, class_num=args.class_num, select_dir=select_dir)
    trainset_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # create model
    checkpoint_path = join(os.getcwd(), 'models', 'v3', 'model_GCRR_autoencoder.pth')
    if args.net == 'autoencoder':
        model = autoencoder().cuda(args.gpu_id)

    load_checkpoint(checkpoint_path, model)
    
    # feature extractor
    feature_extractor(model, trainset_loader, args)
