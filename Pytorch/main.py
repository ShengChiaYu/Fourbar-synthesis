import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import glob
import os
import sys
import argparse
from os.path import join

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import FDs
from models import Net_1
from utils import AverageMeter

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Fourbar network')
    # Load data
    parser.add_argument('--data_dir', dest='data_dir',
                      help='directory to dataset', default="data",
                      type=str)
    parser.add_argument('--pos_num', dest='pos_num',
                        help='60, 360',
                        default='60positions', type=str)
    parser.add_argument('--target', dest='tar',
                        help='param, pos',
                        default='_param.csv', type=str)
    # Training setup
    parser.add_argument('--net', dest='net',
                        help='Net_1',
                        default='Net_1', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=100, type=int)
    parser.add_argument('--loss', dest='loss',
                        help='which loss function to use', default="MSELoss",
                        type=str)
    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="models",
                        type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=6, type=int)
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=16, type=int)
    parser.add_argument('--cuda', dest='use_cuda',
                        help='whether use CUDA',
                        default=False, type=bool)
    parser.add_argument('--gpu', dest='gpu_id',
                        default=0, type=int,
    				    help='GPU id to use.')

    # Configure optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.001, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=5, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)
    args = parser.parse_args()

    return args

def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

def train(model, trainset_loader, valset_loader, args):
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    if args.loss == 'MSELoss':
        criterion = nn.MSELoss()
    # train
    model.train()

    f = open('result.txt', 'w')
    for epoch in range(args.start_epoch, args.max_epochs + 1):
        model.train()

        # training
        print ('Epoch = {}'.format(epoch))
        train_losses = AverageMeter()
        train_pbar = tqdm(total=len(trainset_loader), ncols=100, leave=True)
        for batch_idx, (inputs, targets) in enumerate(trainset_loader):
            if args.use_cuda:
                inputs, targets = inputs.cuda(args.gpu_id), targets.cuda(args.gpu_id)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            train_losses.update(loss.data.item(), inputs.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_pbar.update()
            if batch_idx % 2 == 0:
                train_pbar.set_postfix({'loss':'{:.4f}'.format(train_losses.avg)})
        train_pbar.set_postfix({'loss':'{:.4f}'.format(train_losses.avg)})
        train_pbar.close()

        # evaluate
        model.eval()
        val_losses = AverageMeter()
        val_pbar = tqdm(total=len(valset_loader), ncols=100, leave=True)
        for batch_idx, (inputs, targets) in enumerate(valset_loader):
            if args.use_cuda:
                inputs, targets = inputs.cuda(args.gpu_id), targets.cuda(args.gpu_id)

            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_losses.update(loss.data.item(), inputs.size(0))

                val_pbar.update()
                if batch_idx % 2 == 0:
                    val_pbar.set_postfix({'loss':'{:.4f}'.format(val_losses.avg)})
        val_pbar.set_postfix({'loss':'{:.4f}'.format(val_losses.avg)})
        val_pbar.close()

        if epoch % 5 == 0:
            save_checkpoint(join(args.save_dir, 'model_{}.pth'.format(epoch)), model, optimizer)

        f.write('Epoch={:3d}, train_loss={:.4f}, val_loss={:.4f}\n'.format(epoch, train_losses.avg, val_losses.avg))
        f.flush()

if __name__ == '__main__':
    # parse args
    args = parse_args()

    args.use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(args.gpu_id) if args.use_cuda else "cpu")
    print(device)

    root = os.path.join(args.data_dir, args.pos_num)
    print('root:', root, ', y file:', args.tar)
    trainset = FDs(root=root, x_file='x_train.csv', y_file='y_train'+args.tar, transform=True)
    valset = FDs(root=root, x_file='x_test.csv', y_file='y_test'+args.tar, transform=True)
    trainset_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valset_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # create model
    model = Net_1()
    if args.use_cuda:
        model = model.cuda(args.gpu_id)

    # start training
    train(model, trainset_loader, valset_loader, args)
