import os
import sys
import glob
import pandas as pd
from os.path import join, splitext
from PIL import Image
import cv2

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from utils import parse_args

class FDs(Dataset):
    max = 1
    min = -1
    def __init__(self, root, x_file, y_file, transform):
        """ Intialize the Fourier descriptors dataset """
        self.root = root
        self.x_file = x_file
        self.y_file = y_file
        self.x_data = []
        self.y_data = []
        self.x_scaler = None
        self.y_scaler = None
        self.x_std = None
        self.y_std = None
        self.transform = transform

        # read files to get scalers
        fx = open(os.path.join(self.root, self.x_file), 'r')
        fy = open(os.path.join(self.root, self.y_file), 'r')
        lines = fx.readlines()
        for line in lines:
            tmp = [float(i) for i in line.split(',')]
            self.x_data.append(tmp)
        lines = fy.readlines()
        for line in lines:
            tmp = [float(i) for i in line.split(',')]
            self.y_data.append(tmp)
        self.x_data = torch.Tensor(self.x_data)
        self.y_data = torch.Tensor(self.y_data)

        if self.transform:
            self.x_scaler = (torch.max(self.x_data,0)[0], torch.min(self.x_data,0)[0])
            # self.y_scaler = (torch.max(self.y_data,0)[0], torch.min(self.y_data,0)[0])

            self.x_std = (self.x_data - self.x_scaler[1]) / (self.x_scaler[0] - self.x_scaler[1])
            # self.y_std = (self.y_data - self.y_scaler[1]) / (self.y_scaler[0] - self.y_scaler[1])

            self.x_data = self.x_std * (self.max - self.min) + self.min
            # self.y_data = self.y_std * (self.max - self.min) + self.min

        self.len = self.x_data.size(0)

    def __getitem__(self, index):
        """ Get a sample from the dataset """

        return self.x_data[index], self.y_data[index]

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len


class image_data(Dataset):
    slices = [[1,2,3,5,6],[0,2,3,5,6],[0,1,3,5,6],[0,1,2,5,6]]
    def __init__(self, root, transform, sol='1', class_num=4, select_dir=None):
        """ Intialize the Fourier descriptors dataset """
        self.root = root
        self.transform = transform
        self.data_sol_1 = []
        self.data_sol_2 = []
        self.sol = sol
        self.class_num = class_num
        self.select_dir = select_dir

        dirs = sorted(glob.glob(join(self.root, '**')))
        if self.select_dir is not None:
            dirs = [dirs[self.select_dir]]
            # slice = self.slices[self.select_dir]

        for i, dir in enumerate(dirs):
            print(dir)
            sub_dirs = sorted(glob.glob(join(dir, '**/')))
            label_csvs = sorted(glob.glob(join(dir, '*.csv')))
            for j, (sub_dir, label_csv) in enumerate(zip(sub_dirs, label_csvs)):
                # print(sub_dir)
                # print(label_csv)
                images = sorted(glob.glob(join(sub_dir, '*.png')))
                params = open(label_csv, 'r').readlines()
                for k in range(len(images)):
                    num = int(splitext(images[k].split('/')[-1])[0]) - 1
                    tmp = [float(i) for i in params[num].split(',')]
                    # if self.select_dir is not None:
                    #     tmp = [tmp[i] for i in slice]

                    data = (num+1, images[k], i, i*3+j, tmp)
                    if num < len(images)/2:
                        self.data_sol_1.append(data)
                    else:
                        self.data_sol_2.append(data)

        if self.sol == 'all':
            self.len = len(self.data_sol_1 + self.data_sol_2)
        else:
            self.len = len(self.data_sol_1)
        print('Length of set: '+ str(self.len))

    def __getitem__(self, index):
        """ Get a sample from the dataset """
        if self.sol == 'all':
            data = (self.data_sol_1 + self.data_sol_2)[index]
        elif self.sol == '1':
            data = self.data_sol_1[index]
        elif self.sol == '2':
            data = self.data_sol_2[index]

        num = data[0]
        image_fn = data[1]
        image = Image.open(image_fn)

        if self.transform is not None:
            image = self.transform(image)

        if self.class_num == 4:
            label = data[2]
        elif self.class_num == 12:
            label = data[3]

        params = torch.Tensor(data[4])

        return num, image, label, params

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len


class image_data_2(Dataset):
    image_size = 28
    def __init__(self, root, transform, class_num=4, select_dir=None):
        """ Intialize the Fourier descriptors dataset """
        self.root = root
        self.transform = transform
        self.data_sol_1 = []
        self.class_num = class_num
        self.select_dir = select_dir

        dirs = sorted(glob.glob(join(self.root, '**')))
        if self.select_dir is not None:
            dirs = [dirs[self.select_dir]]

        for i, dir in enumerate(dirs):
            print(dir)
            sub_dirs = sorted(glob.glob(join(dir, '**/')))
            label_csvs = sorted(glob.glob(join(dir, '*.csv')))
            for j, (sub_dir, label_csv) in enumerate(zip(sub_dirs, label_csvs)):
                # print(sub_dir)
                # print(label_csv)
                images = sorted(glob.glob(join(sub_dir, '*.png')))
                params = open(label_csv, 'r').readlines()
                for k in range(len(images)):
                    num = int(splitext(images[k].split('/')[-1])[0]) - 1
                    tmp = [float(i) for i in params[num].split(',')]

                    data = (num+1, images[k], i, i*3+j, tmp)
                    self.data_sol_1.append(data)

        self.len = len(self.data_sol_1)
        print('Length of set: '+ str(self.len))

    def __getitem__(self, index):
        """ Get a sample from the dataset """
        data = self.data_sol_1[index]

        image_fn = data[1]
        image = cv2.imread(image_fn, cv2.IMREAD_GRAYSCALE)
        # image = cv2.resize(image,(self.image_size,self.image_size))

        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len


class image_data_3(Dataset):
    image_size = 224
    slices = [[1,2,3,5,6],[0,2,3,5,6],[0,1,3,5,6],[0,1,2,5,6]]
    def __init__(self, root, transform, class_num=4, select_dir=None):
        """ Intialize the Fourier descriptors dataset """
        self.root = root
        self.transform = transform
        self.data_sol_1 = []
        self.class_num = class_num
        self.select_dir = select_dir

        dirs = sorted(glob.glob(join(self.root, '**')))
        if self.select_dir is not None:
            dirs = [dirs[self.select_dir]]
            slice = self.slices[self.select_dir]

        for i, dir in enumerate(dirs):
            print(dir)
            sub_dirs = sorted(glob.glob(join(dir, '**/')))
            label_csvs = sorted(glob.glob(join(dir, '*.csv')))
            for j, (sub_dir, label_csv) in enumerate(zip(sub_dirs, label_csvs)):
                print(sub_dir)
                print(label_csv)
                part_dirs = sorted(glob.glob(join(sub_dir, '**/')))
                params = open(label_csv, 'r').readlines()
                for part_dir in part_dirs:
                    # print(part_dir)
                    images = sorted(glob.glob(join(part_dir, '*.png')))
                    for k in range(len(images)):
                        num = int(splitext(images[k].split('/')[-1])[0]) - 1
                        tmp = [float(i) for i in params[num].split(',')[:-1]]
                        tmp = [tmp[i] for i in slice]

                        data = (num+1, images[k], i, i*3+j, tmp)
                        self.data_sol_1.append(data)

        self.len = len(self.data_sol_1)
        print('Length of set: '+ str(self.len))

    def __getitem__(self, index):
        """ Get a sample from the dataset """
        data = self.data_sol_1[index]

        num = data[0]
        image_fn = data[1]
        # image = cv2.imread(image_fn, cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(image_fn)
        image = cv2.resize(image,(self.image_size,self.image_size))
        # image = Image.open(image_fn)

        if self.transform is not None:
            image = self.transform(image)

        if self.class_num == 4:
            label = data[2]
        elif self.class_num == 12:
            label = data[3]

        params = torch.Tensor(data[4])

        return num, image, label, params

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len


def test():
    root = os.path.join('data','60positions')
    y_file = '_param.csv'
    print('root:', root, ', y file:', y_file)

    testset = FDs(root=root, x_file='x_test.csv', y_file='y_test'+y_file, transform=True)
    inputs, targets = testset.__getitem__(9098) # max(9098,1) min(9355,1)
    print(inputs)
    print(targets)


def test_image_data():
    # parse args
    args = parse_args()

    root = join(os.getcwd(), '../Pytorch', 'image_data', 'train')
    print('root:', root)
    transform=transforms.Compose([
              # transforms.RandomCrop(args.img_size, padding=None, pad_if_needed=True, fill=0, padding_mode='edge'),
              transforms.Resize((args.img_size,args.img_size)),
              transforms.ToTensor(),
              # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dirs = {'GCCC':0, 'GCRR':1,'GRCR':2,'GRRC':3}
    select_dir = dirs[args.select_dir]
    dataset = image_data(root=root, transform=transform, sol='all', class_num=args.class_num, select_dir=select_dir)

    # test one sample
    num, image, label, params = dataset.__getitem__(2)
    # image.show()
    print(num)
    # print(image.shape)
    # print(image[1][50])
    # print(label)
    # print(params)

    # test a batch
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
    #                                          shuffle=True, num_workers=6)
    # dataiter = iter(dataloader)
    # images, labels, targets = dataiter.next()
    #
    # print('Image tensor in each batch:', images.shape, images.dtype)
    # print('Label tensor in each batch:', labels.shape, labels.dtype)
    # print('Target tensor in each batch:', targets.shape, targets.dtype)
    # print(images)
    # print(labels)
    pass


def test_image_data_2():
    # parse args
    args = parse_args()

    root = join(os.getcwd(), '../Pytorch', 'image_data_2', 'train')
    print('root:', root)
    transform=transforms.Compose([
              # transforms.RandomCrop(args.img_size, padding=None, pad_if_needed=True, fill=0, padding_mode='edge'),
              transforms.ToTensor(),
              # transforms.Resize((args.img_size,args.img_size)),
              # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dirs = {'GCCC':0, 'GCRR':1,'GRCR':2,'GRRC':3}
    select_dir = dirs[args.select_dir]
    dataset = image_data_2(root=root, transform=None, class_num=args.class_num, select_dir=select_dir)

    # test one sample
    num, image, label, params = dataset.__getitem__(19655)
    cv2.imshow('Test', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(num)
    print(image.shape)
    # print(image[1][50])
    print(label)
    print(params)
    pass


def test_image_data_3():
    # parse args
    args = parse_args()

    root = join(os.getcwd(), '../Encoder/image_data_3')
    print('root:', root)
    transform=transforms.Compose([
              transforms.RandomCrop((args.img_size,args.img_size), padding=None, pad_if_needed=False, fill=0),
              # transforms.Resize((args.img_size,args.img_size)),
              # transforms.ToTensor(),
              # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dirs = {'GCCC':0, 'GCRR':1,'GRCR':2,'GRRC':3}
    select_dir = dirs[args.select_dir]
    dataset = image_data_3(root=root, transform=transform, class_num=args.class_num, select_dir=select_dir)

    # test one sample
    num, image, label, params = dataset.__getitem__(1)
    # image = dataset.__getitem__(19000)
    # cv2.imshow('Test', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    image.show()
    print(num)
    print(image.size)
    print(label)
    print(params)
    pass


if __name__ == '__main__':
    # test()
    # test_image_data()
    # test_image_data_2()
    test_image_data_3()
