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

from dataset import image_data
from models import LeNet, resnet50, resnet152
from inception import inception_v3
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


def test(valset_loader, args, criterion, aug):
    model.eval()
    val_losses, val_acc = AverageMeter(aug), AverageMeter(aug)

    val_pbar = tqdm(total=len(valset_loader), ncols=100, leave=True)
    for batch_idx, (inputs, labels, params) in enumerate(valset_loader):
        if args.select_dir is not None:
            targets = params
        else:
            targets = labels

        if args.use_cuda:
            inputs, targets = inputs.cuda(args.gpu_id), targets.cuda(args.gpu_id)

        with torch.no_grad():
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            val_losses.update(loss.data.item(), inputs.shape[0])

            if args.select_dir is not None:
                diff = abs((outputs-targets)/targets)
                acc = torch.numel(diff[diff<args.threshold]) / float(torch.numel(diff))
                val_acc.update(acc, torch.numel(diff))
            else:
                _, pred = torch.max(outputs, 1)
                acc = sum(pred == targets).float() / float(pred.shape[0])
                val_acc.update(acc, pred.shape[0])

            val_pbar.update()

            val_pbar.set_postfix({'loss':'{:.4f}'.format(val_losses.avg),
                                    'acc':'{:.4f}'.format(val_acc.avg),
                                    })
    val_pbar.close()

    return val_losses, val_acc


def train(model, trainset_loader, valset_loader, args):
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    if args.select_dir is not None:
        criterion = nn.MSELoss()
        log = open(join(args.save_dir, 'loss_{}_{}.txt'.format(args.select_dir, args.net)), 'w')
    else:
        criterion = nn.CrossEntropyLoss()
        log = open(join(args.save_dir, 'loss_{}_{}.txt'.format(args.class_num, args.net)), 'w')

    # train
    model.train()
    best_acc = 0
    for epoch in range(args.start_epoch, args.max_epochs + 1):
        # training
        model.train()

        print ('\nEpoch = {}'.format(epoch))
        log.write('Epoch = {}, '.format(epoch))
        aug = 1 # augmentation for loss
        train_losses, train_acc = AverageMeter(aug), AverageMeter(aug)

        train_pbar = tqdm(total=len(trainset_loader), ncols=100, leave=True)
        for batch_idx, (inputs, labels, params) in enumerate(trainset_loader):
            if args.select_dir is not None:
                targets = params
            else:
                targets = labels

            if args.use_cuda:
                inputs, targets = inputs.cuda(args.gpu_id), targets.cuda(args.gpu_id)

            if args.net == 'inception_v3':
                outputs, aux_outputs = model(inputs)
                loss1 = criterion(outputs, targets)
                loss2 = criterion(aux_outputs, targets)
                loss = loss1 + 0.4*loss2
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            train_losses.update(loss.data.item(), inputs.shape[0])

            if args.select_dir is not None:
                diff = abs(outputs-targets) / targets
                acc = torch.numel(diff[diff<args.threshold]) / float(torch.numel(diff))
                train_acc.update(acc, torch.numel(diff))
            else:
                _, pred = torch.max(outputs, 1)
                acc = sum(pred == targets).float() / float(pred.shape[0])
                train_acc.update(acc, pred.shape[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_pbar.update()

            train_pbar.set_postfix({'loss':'{:.4f}'.format(train_losses.avg),
                                    'acc':'{:.4f}'.format(train_acc.avg),
                                    })

        train_pbar.close()

        # evaluate
        val_losses, val_acc = test(valset_loader, args, criterion, aug)

        if val_acc.avg > best_acc:
            best_acc = val_acc.avg
            if args.select_dir is not None:
                save_checkpoint(join(args.save_dir, 'model_{}_{}.pth'.format(args.select_dir, args.net)), model, optimizer)
            else:
                save_checkpoint(join(args.save_dir, 'model_{}_{}.pth'.format(args.class_num, args.net)), model, optimizer)

        log.write('Loss: {:.4f}/{:.4f}, Accuracy: {:.4f}/{:.4f}\n'.format(train_losses.avg, val_losses.avg,
                      train_acc.avg, val_acc.avg))
        log.flush()

    log.write('Best test_acc: {:.4f}\n'.format(best_acc))
    log.close()

if __name__ == '__main__':
    # parse args
    args = parse_args()

    args.use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(args.gpu_id) if args.use_cuda else "cpu")
    print(device)

    root = join(os.getcwd(), 'image_data')
    print('root:', root)

    if args.select_dir is not None:
        dirs = {'GCCC':0, 'GCRR':1,'GRCR':2,'GRRC':3}
        select_dir = dirs[args.select_dir]
    else:
        select_dir = None

    transform=transforms.Compose([
              # transforms.RandomCrop(args.img_size, padding=None, pad_if_needed=True, fill=0, padding_mode='edge'),
              transforms.Resize((args.img_size,args.img_size)),
              transforms.ToTensor(),
              # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = image_data(root=join(root,'train'), transform=transform, sol='1', class_num=args.class_num, select_dir=select_dir)
    valset = image_data(root=join(root,'valid'), transform=transform, sol='1', class_num=args.class_num, select_dir=select_dir)

    trainset_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valset_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # create model
    if args.net == 'LeNet':
        model = LeNet(args).cuda(args.gpu_id)
    elif args.net == 'resnet50':
        model = resnet50(args).cuda(args.gpu_id)
    elif args.net == 'resnet152':
        model = resnet152(args).cuda(args.gpu_id)
    elif args.net == 'inception_v3':
        model = inception_v3(args).cuda(args.gpu_id)


    # start training
    train(model, trainset_loader, valset_loader, args)
