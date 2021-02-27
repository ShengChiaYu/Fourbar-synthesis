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

from dataset import FDs
from model import Net_1, Net_2, Net_3, Net_4
from utils import AverageMeter, parse_args
from preprocessing import matching
from test import test

testing_data_name = [
'Banana','Crescent','Double_straight',
'Figure_eight','Kidney_bean',
'Scimitar','Single_straight','Teardrop',
'Triple_cusps','Triple_loops','Umbrella']

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
    criterion = nn.MSELoss()

    batch_len = len(valset_loader)
    val_pbar = tqdm(total=batch_len, ncols=10, leave=True)

    val_img_num = 5
    sample_batch = [int(batch_len*i/val_img_num) for i in range(val_img_num)]

    val_losses, val_mde = AverageMeter(), AverageMeter()
    for batch_idx, (inputs,links,p1s) in enumerate(valset_loader):
        if args.use_cuda:
            inputs, links = inputs.cuda(args.gpu_id), links.cuda(args.gpu_id)
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, links)
            val_losses.update(loss.data.item(), inputs.shape[0])

            p1s = p1s.cpu().numpy()
            outputs = outputs.cpu().numpy()
            if (epoch==1 or epoch%10==0) and batch_idx in sample_batch:
                plot = True
            else:
                plot = False

            mde = matching(epoch, batch_idx, outputs, p1s, testing_data_name, args, plot=plot)

            val_mde.update(mde, inputs.shape[0])

            val_pbar.update()
            val_pbar.set_postfix({'loss':'{:.4f}'.format(val_losses.avg),
                                  'mde':'{:.4f}'.format(val_mde.avg),
                                  })

    val_pbar.close()

    return val_losses, val_mde


def train(model, trainset_loader, valset_loader, train_num, args):
    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=1e-5)


    criterion = nn.MSELoss()
    if '_b_' in args.dataset:
        args.save_dir = os.path.join(os.getcwd(),args.save_dir,'model_{}_data_b_1215_{}_{}'.format(args.select_dir,train_num,args.n_clusters))
        log = open(join(args.save_dir, 'loss_{}_{}_{}.txt'.
              format(args.select_dir, args.dataset, args.n_clusters)), 'w')
        best_model_name = 'model_{}_{}_{}_best.pth'.format(args.select_dir, args.dataset, args.n_clusters)
        model_name = 'model_{}_{}_{}.pth'.format(args.select_dir, args.dataset, args.n_clusters)
    else:
        log = open(join(args.save_dir, 'loss_{}_{}.txt'.format(args.select_dir, args.dataset)), 'w')
        best_model_name = 'model_{}_{}_best.pth'.format(args.select_dir, args.dataset)
        model_name = 'model_{}_{}.pth'.format(args.select_dir, args.dataset)

    # train
    best_mde = 100
    early_stopping = 0
    log.write('Ep, T_loss, V_loss, Valmde\n')
    for epoch in range(args.start_epoch, args.max_epochs + 1):
        # training
        model.train()

        print ('\nEpoch = {}'.format(epoch))
        log.write('{}, '.format(epoch))
        train_losses =  AverageMeter()

        train_pbar = tqdm(total=len(trainset_loader), ncols=10, leave=True)
        for batch_idx, (inputs,targets,_) in enumerate(trainset_loader):
            optimizer.zero_grad()

            if args.use_cuda:
                inputs, targets = inputs.cuda(args.gpu_id), targets.cuda(args.gpu_id)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            train_losses.update(loss.data.item(), inputs.shape[0])

            loss.backward()
            optimizer.step()
            train_pbar.update()

            train_pbar.set_postfix({'loss':'{:.4f}'.format(train_losses.avg)})

        train_pbar.close()

        # evaluate
        val_losses, val_mde = valid(model, epoch, valset_loader, args)
        # test_mde = test(model, epoch, testset_loader, best_mde, args)

        if val_mde.avg < best_mde:
            best_mde = val_mde.avg
            early_stopping = 0
            save_checkpoint(join(args.save_dir, best_model_name), model, optimizer)
        else:
            early_stopping += 1

        save_checkpoint(join(args.save_dir, model_name), model, optimizer)
        print('Early stopping: {}'.format(early_stopping))

        log.write('{:.4f}, {:.4f}, {:.4f}\n'.
                  format(train_losses.avg, val_losses.avg, val_mde.avg))
        log.flush()

        if early_stopping >= args.patience:
            print('Early stopped.')
            break

    log.write('Best test_acc: {:.4f}\n'.format(best_mde))
    log.close()


if __name__ == '__main__':
    # parse args
    args = parse_args()

    args.use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(args.gpu_id) if args.use_cuda else "cpu")
    print(device)

    if 'G' in args.select_dir:
        train_num = 14560
    else:
        train_num = 12896

    if '_b_' in args.dataset:
        save_path = os.path.join(os.getcwd(),args.save_dir,'model_{}_data_b_1215_{}_{}'.format(args.select_dir,train_num,args.n_clusters))
        try:
            os.makedirs(save_path)
            print("Directory ", save_path, " Creates.")

        except FileExistsError:
            print("Directory ", save_path, " already exists.")


    if '_b_' in args.dataset:
        train_data_name = '{}_{}_{}.joblib'.format(args.dataset,train_num,args.n_clusters)
        test_data_name = '{}_2496_{}.joblib'.format(args.dataset,args.n_clusters)
    else:
        train_data_name = '{}_{}.joblib'.format(args.dataset,train_num)
        test_data_name = '{}_2496.joblib'.format(args.dataset)

    train_root = os.path.join(args.data_dir, train_data_name)
    val_root = os.path.join(args.data_dir, test_data_name)
    # test_root = os.path.join(args.data_dir, 'testing_data.joblib')
    print('train root:', train_root)
    print('valid root:', val_root)
    # print('testing root:', test_root)

    trainset = FDs(root=train_root, select_dir=args.select_dir)
    valset = FDs(root=val_root, scale=trainset.scale, select_dir=args.select_dir, valid=True)
    # testset = FDs(root=test_root, scale=trainset.scale, train=False)

    trainset_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valset_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    # testset_loader = DataLoader(testset, batch_size=11, shuffle=False, num_workers=args.num_workers)

    # create model
    if args.net == 'Net_1':
        model = Net_1().double().cuda(args.gpu_id)
    elif args.net == 'Net_2':
        model = Net_2().double().cuda(args.gpu_id)
    elif args.net == 'Net_3':
        model = Net_3().double().cuda(args.gpu_id)
    elif args.net == 'Net_4':
        model = Net_4().double().cuda(args.gpu_id)

    # start training
    train(model, trainset_loader, valset_loader, train_num, args)
