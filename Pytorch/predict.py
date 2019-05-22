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
from tqdm import tqdm

from dataset import FDs
from models import Net_1
from utils import AverageMeter, parse_args


def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

def predict(model, valset_loader, args):
    if args.tar == '_pos.csv':
        if args.pos_num == '60positions':
            criterion = PosLoss(60)
        if args.pos_num == '360positions':
            criterion = PosLoss(360)
    else:
        criterion = nn.MSELoss()

    model.eval()

    aug = 1 # augmentation for loss
    val_losses = AverageMeter(aug)
    if args.tar == '_pos.csv':
        invalid_losses, up_losses, low_losses, bo_losses, no_losses = AverageMeter(aug), AverageMeter(aug), \
        AverageMeter(aug), AverageMeter(aug), AverageMeter(aug)

    val_pbar = tqdm(total=len(valset_loader), ncols=100, leave=True)
    filepath = os.path.join(os.getcwd(), 'result', '{}.csv'.format(args.model_name))
    file = open(filepath, 'w')
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

            for i in range(args.batch_size):
                file.write('{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(outputs[i,0].item(), outputs[i,1].item(),
                outputs[i,2].item(), outputs[i,3].item(), outputs[i,4].item()))

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
    file.close()

if __name__ == '__main__':
    # parse args
    args = parse_args()

    args.use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(args.gpu_id) if args.use_cuda else "cpu")
    print(device)

    root = os.path.join(args.data_dir, args.pos_num)
    print('root:', root, ', y file:', args.tar)
    valset = FDs(root=root, x_file='x_test.csv', y_file='y_test'+args.tar, transform=True)
    valset_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # create model
    model = Net_1().cuda(args.gpu_id)
    checkpoint_path = os.path.join(os.getcwd(), args.model_dir, '{}.pth'.format(args.model_name))
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    load_checkpoint(checkpoint_path, model, optimizer)

    # start training
    predict(model, valset_loader, args)
