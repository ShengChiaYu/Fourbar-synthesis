import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import os
import sys
import argparse
from os.path import join

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import multi_FDs
from model import classifier
from utils import AverageMeter, parse_args


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


def valid(model, epoch, valset_loader, args):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    batch_len = len(valset_loader)
    val_pbar = tqdm(total=batch_len, ncols=10, leave=True)

    val_losses, val_acc = AverageMeter(), AverageMeter()
    for batch_idx, (inputs,targets) in enumerate(valset_loader):

        if args.use_cuda:
            inputs, targets = inputs.cuda(args.gpu_id), targets.cuda(args.gpu_id)

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            _, pred = torch.max(outputs, 1)
            correct = sum(pred == targets).float() / args.batch_size

            val_losses.update(loss.data.item(), args.batch_size)
            val_acc.update(correct.data.item(), args.batch_size)

            val_pbar.update()
            val_pbar.set_postfix({'loss':'{:.4f}'.format(val_losses.avg),
                                   'acc':'{:.4f}'.format(val_acc.avg),
                                 })

    val_pbar.close()

    return val_losses, val_acc


def train(model, trainset_loader, valset_loader, args):
    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=1e-5)


    criterion = nn.CrossEntropyLoss()
    log = open(join(args.save_dir, 'loss_cls.txt'), 'w')
    best_model_name = 'model_cls_best.pth'
    model_name = 'model_cls.pth'

    # train
    best_acc = 0
    early_stopping = 0
    log.write('Ep, t_loss, t_acc, v_loss, v_acc\n')
    for epoch in range(args.start_epoch, args.max_epochs + 1):
        # training
        model.train()

        print ('\nEpoch = {}'.format(epoch))
        log.write('{}, '.format(epoch))
        train_losses =  AverageMeter()
        train_acc =  AverageMeter()

        train_pbar = tqdm(total=len(trainset_loader), ncols=10, leave=True)
        for batch_idx, (inputs,targets) in enumerate(trainset_loader):
            optimizer.zero_grad()

            if args.use_cuda:
                inputs, targets = inputs.cuda(args.gpu_id), targets.cuda(args.gpu_id)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            _, pred = torch.max(outputs, 1)
            correct = sum(pred == targets).float() / args.batch_size

            train_losses.update(loss.data.item(), args.batch_size)
            train_acc.update(correct.data.item(), args.batch_size)

            loss.backward()
            optimizer.step()
            train_pbar.update()

            train_pbar.set_postfix({'loss':'{:.4f}'.format(train_losses.avg),
                                    'acc':'{:.4f}'.format(train_acc.avg),
                                    })

        train_pbar.close()

        # evaluate
        val_losses, val_acc = valid(model, epoch, valset_loader, args)

        if val_acc.avg > best_acc:
            best_acc = val_acc.avg
            early_stopping = 0
            save_checkpoint(join(args.save_dir, best_model_name), model, optimizer)
        else:
            early_stopping += 1

        save_checkpoint(join(args.save_dir, model_name), model, optimizer)
        print('Early stopping: {}'.format(early_stopping))

        log.write('{:.4f}, {:.4f}, {:.4f}, {:.4f}\n'.
                  format(train_losses.avg, train_acc.avg, val_losses.avg, val_acc.avg))
        log.flush()

        if early_stopping >= args.patience:
            print('Early stopped.')
            break

    log.write('Best test_acc: {:.4f}\n'.format(best_acc))
    log.close()


if __name__ == '__main__':
    # parse args
    args = parse_args()

    args.use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(args.gpu_id) if args.use_cuda else "cpu")
    print(device)

    trainset = multi_FDs(root=args.data_dir, train=True)
    valset = multi_FDs(root=args.data_dir, train=False)

    trainset_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valset_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # create model
    if args.net == 'classifier':
        model = classifier().double().cuda(args.gpu_id)

    # start training
    train(model, trainset_loader, valset_loader, args)
