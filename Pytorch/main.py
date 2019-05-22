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
from utils import AverageMeter, parse_args
from loss import PosLoss


def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)


def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)


def test(valset_loader, args, criterion, aug):
    model.eval()
    val_losses = AverageMeter(aug)
    if args.tar == '_pos.csv':
        invalid_losses, up_losses, low_losses, bo_losses, no_losses = AverageMeter(aug), AverageMeter(aug), \
        AverageMeter(aug), AverageMeter(aug), AverageMeter(aug)

    val_pbar = tqdm(total=len(valset_loader), ncols=100, leave=True)
    for batch_idx, (inputs, targets) in enumerate(valset_loader):
        if args.use_cuda:
            inputs, targets = inputs.cuda(args.gpu_id), targets.cuda(args.gpu_id)

        with torch.no_grad():
            outputs = model(inputs)
            if args.tar == '_pos.csv':
                invalid_loss, up_loss, low_loss, bo_loss, no_loss = criterion(outputs, targets)

                invalid_losses.update(invalid_loss.data.item(), inputs.size(0))
                up_losses.update(up_loss.data.item(), inputs.size(0))
                low_losses.update(low_loss.data.item(), inputs.size(0))
                bo_losses.update(bo_loss.data.item(), inputs.size(0))
                no_losses.update(no_loss.data.item(), inputs.size(0))

                loss = invalid_loss + up_loss + low_loss + bo_loss + no_loss
            else:
                loss = criterion(outputs, targets)

            val_losses.update(loss.data.item(), inputs.size(0))

            val_pbar.update()
            if batch_idx % 2 == 0:
                if args.tar == '_pos.csv':
                    val_pbar.set_postfix({'loss':'{:.4f}'.format(val_losses.avg), 'inv':'{:.4f}'.format(invalid_losses.avg),
                                            'up':'{:.4f}'.format(up_losses.avg), 'low':'{:.4f}'.format(low_losses.avg),
                                            'bo':'{:.4f}'.format(bo_losses.avg), 'no_':'{:.4f}'.format(no_losses.avg)})
                else:
                    val_pbar.set_postfix({'loss':'{:.4f}'.format(val_losses.avg)})
    if args.tar == '_pos.csv':
        val_pbar.set_postfix({'loss':'{:.4f}'.format(val_losses.avg), 'inv':'{:.4f}'.format(invalid_losses.avg),
                                'up':'{:.4f}'.format(up_losses.avg), 'low':'{:.4f}'.format(low_losses.avg),
                                'bo':'{:.4f}'.format(bo_losses.avg), 'no_':'{:.4f}'.format(no_losses.avg)})
    else:
        val_pbar.set_postfix({'loss':'{:.4f}'.format(val_losses.avg)})
    val_pbar.close()

    return val_losses


def train(model, trainset_loader, valset_loader, args):
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    if args.tar == '_pos.csv':
        if args.pos_num == '60positions':
            criterion = PosLoss(60)
        if args.pos_num == '360positions':
            criterion = PosLoss(360)
    else:
        criterion = nn.MSELoss()

    # train
    model.train()

    f = open('loss.txt', 'w')
    for epoch in range(args.start_epoch, args.max_epochs + 1):
        # training
        model.train()

        print ('Epoch = {}'.format(epoch))
        aug = 1 # augmentation for loss
        train_losses = AverageMeter(aug)

        if args.tar == '_pos.csv':
            invalid_losses, up_losses, low_losses, bo_losses, no_losses = AverageMeter(aug), AverageMeter(aug), \
            AverageMeter(aug), AverageMeter(aug), AverageMeter(aug)

        train_pbar = tqdm(total=len(trainset_loader), ncols=100, leave=True)
        for batch_idx, (inputs, targets) in enumerate(trainset_loader):
            if args.use_cuda:
                inputs, targets = inputs.cuda(args.gpu_id), targets.cuda(args.gpu_id)

            outputs = model(inputs)

            if args.tar == '_pos.csv':
                invalid_loss, up_loss, low_loss, bo_loss, no_loss = criterion(outputs, targets)

                invalid_losses.update(invalid_loss.data.item(), inputs.size(0))
                up_losses.update(up_loss.data.item(), inputs.size(0))
                low_losses.update(low_loss.data.item(), inputs.size(0))
                bo_losses.update(bo_loss.data.item(), inputs.size(0))
                no_losses.update(no_loss.data.item(), inputs.size(0))

                loss = invalid_loss + up_loss + low_loss + bo_loss + no_loss
            else:
                loss = criterion(outputs, targets)

            train_losses.update(loss.data.item(), inputs.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_pbar.update()
            if batch_idx % 2 == 0:
                if args.tar == '_pos.csv':
                    train_pbar.set_postfix({'loss':'{:.4f}'.format(train_losses.avg), 'inv':'{:.4f}'.format(invalid_losses.avg),
                                            'up':'{:.4f}'.format(up_losses.avg), 'low':'{:.4f}'.format(low_losses.avg),
                                            'bo':'{:.4f}'.format(bo_losses.avg), 'no_':'{:.4f}'.format(no_losses.avg)})
                else:
                    train_pbar.set_postfix({'loss':'{:.4f}'.format(train_losses.avg)})
        if args.tar == '_pos.csv':
            train_pbar.set_postfix({'loss':'{:.4f}'.format(train_losses.avg), 'inv':'{:.4f}'.format(invalid_losses.avg),
                                    'up':'{:.4f}'.format(up_losses.avg), 'low':'{:.4f}'.format(low_losses.avg),
                                    'bo':'{:.4f}'.format(bo_losses.avg), 'no_':'{:.4f}'.format(no_losses.avg)})
        else:
            train_pbar.set_postfix({'loss':'{:.4f}'.format(train_losses.avg)})

        train_pbar.close()

        # evaluate
        val_losses = test(valset_loader, args, criterion, aug)

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
    model = Net_1().cuda(args.gpu_id)
    if args.pretrained:
        checkpoint_path = os.path.join(os.getcwd(), 'models_pretrained', 'model_100.pth')
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        load_checkpoint(checkpoint_path, model, optimizer)

    # start training
    train(model, trainset_loader, valset_loader, args)
