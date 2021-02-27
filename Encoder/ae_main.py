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


def test(model, testset_loader, args, epoch):

    with torch.no_grad():
        # obtain one batch of test images
        dataiter = iter(testset_loader)
        _, images, _, _ = dataiter.next()
        images = images.cuda()

        # get sample outputs
        output = model(images)
        # prep images for display
        images = images.cpu().numpy()

        # output is resized into a batch of iages
        output = output.view(args.batch_size, 1, args.img_size, args.img_size)
        # use detach when it's an output that requires_grad
        output = output.detach().cpu().numpy()

        # plot the first ten input images and then reconstructed images
        fig, axes = plt.subplots(nrows=2, ncols=16, sharex=True, sharey=True, figsize=(25,4))

        # input images on top row, reconstructions on bottom
        for images, row in zip([images, output], axes):
            for img, ax in zip(images, row):
                ax.imshow(np.squeeze(img), cmap='gray')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

        save_path = os.path.join(os.getcwd(), 'verified_images',
                                '{}_{}_{}.jpg'.format(args.select_dir, args.net, epoch))
        plt.savefig(save_path)
        print('output jpg path:{}'.format(save_path))


def train(model, trainset_loader, args):
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=1e-5)

    if args.select_dir is not None:
        criterion = nn.MSELoss()
        log = open(join(args.save_dir, 'loss_{}_{}.txt'.format(args.select_dir, args.net)), 'w')
    else:
        criterion = nn.CrossEntropyLoss()
        log = open(join(args.save_dir, 'loss_{}_{}.txt'.format(args.class_num, args.net)), 'w')

    # train
    for epoch in range(args.start_epoch, args.max_epochs + 1):
        # training
        model.train()

        print ('\nEpoch = {}'.format(epoch))
        log.write('Epoch = {}, '.format(epoch))
        aug = 1 # augmentation for loss
        train_losses = AverageMeter(aug)

        train_pbar = tqdm(total=len(trainset_loader), ncols=10, leave=True)
        for batch_idx, (_, inputs, _, _) in enumerate(trainset_loader):

            if args.use_cuda:
                inputs = inputs.cuda(args.gpu_id)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, inputs)

            train_losses.update(loss.data.item(), inputs.shape[0])

            loss.backward()
            optimizer.step()
            train_pbar.update()

            train_pbar.set_postfix({'loss':'{:.4f}'.format(train_losses.avg)})

        train_pbar.close()

        # evaluate
        test(model, trainset_loader, args, epoch)
        save_checkpoint(join(args.save_dir, 'model_{}_{}.pth'.format(args.select_dir, args.net)), model, optimizer)

        log.write('Loss: {:.4f}\n'.format(train_losses.avg))
        log.flush()

    log.close()

if __name__ == '__main__':
    # parse args
    args = parse_args()

    args.use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(args.gpu_id) if args.use_cuda else "cpu")

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
    trainset_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # create model
    if args.net == 'autoencoder':
        model = autoencoder().cuda(args.gpu_id)

    # start training
    train(model, trainset_loader, args)
