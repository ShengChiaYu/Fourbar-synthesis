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
from joblib import dump, load

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import FDs
from model import Net_1, Net_2, Net_3, Net_4, classifier
from utils import AverageMeter, parse_args
from preprocessing import matching, matching2, matching3

testing_data_name = [
'Banana','Crescent','Double_straight',
'Figure_eight','Kidney_bean',
'Scimitar','Single_straight','Teardrop',
'Triple_cusps','Triple_loops','Umbrella']


def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('=======================================')
    print('model loaded from %s' % checkpoint_path)


def test(model, ep, testset_loader, best_mde, args):
    model.eval()
    test_mde = AverageMeter()

    test_pbar = tqdm(total=len(testset_loader), ncols=10, leave=True)
    for batch_idx, (fns, inputs, targets) in enumerate(testset_loader):

        if args.use_cuda:
            inputs, targets = inputs.cuda(args.gpu_id), targets.cpu().numpy()

        with torch.no_grad():
            L = model(inputs).cpu().numpy()
            mde = matching(ep, batch_idx, L, targets, fns, args, plot=True)

            if mde<best_mde:
                mde = matching(ep, batch_idx, L, targets, fns, args, plot=True)

            test_mde.update(mde, inputs.shape[0])

            test_pbar.update()

            test_pbar.set_postfix({'mde':'{:.4f}'.format(test_mde.avg)})
    test_pbar.close()

    return test_mde


def valid(model, epoch, valset_loader, args):
    model.eval()
    criterion = nn.MSELoss()

    batch_len = len(valset_loader)
    val_pbar = tqdm(total=batch_len, ncols=10, leave=True)

    val_img_num = 3
    sample_batch = [int(batch_len*i/val_img_num) for i in range(val_img_num)]

    val_losses, val_mde = AverageMeter(), AverageMeter()
    for batch_idx, (inputs,links,p1s) in enumerate(valset_loader):
        print(inputs.shape)
        sys.exit()
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


def model_comparison(model, valset, fn, n_clusters, begin, args):
    model.eval()
    criterion = nn.MSELoss()

    # data_nums = {
    # # 'GCCC':[], # 0,8
    # # 'GCRR':[],  # 0
    # # 'GRCR':[],  # 0,3,5
    # # 'GRRC':[],  # 1,3,11
    # 'RRR1':[1,629,1253,1877],    # 4,7
    # 'RRR2':[8,626,1251,1879],
    # 'RRR3':[4,281,560,835,1112,1391,1674,1943,2224],
    # 'RRR4':[0,425,833,1251,1666,2085],
    # }
    step = valset.len / n_clusters
    nums = []
    inputs = []
    links = []
    p1s = []

    for i in range(begin,valset.len,int(step)):
    # for i in data_nums[args.select_dir]:
        nums.append(i)
        input, link, p1 = valset.__getitem__(i)
        inputs.append(input.unsqueeze(0))
        links.append(link.unsqueeze(0))
        p1s.append(p1.unsqueeze(0))
        print(i)
        print(link)
    inputs = torch.cat(inputs,dim=0)
    links = torch.cat(links,dim=0)
    p1s = torch.cat(p1s,dim=0)

    inputs, links = inputs.cuda(args.gpu_id), links.cuda(args.gpu_id)
    with torch.no_grad():
        output = model(inputs)
        loss = criterion(output, links).item()

        p1s = p1s.cpu().numpy()
        output = output.cpu().numpy()

        mde = matching2(nums, output, p1s, fn, args, plot=True)


    return loss, mde


def completed_model_test():
    # parse args
    args = parse_args()

    args.use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(args.gpu_id) if args.use_cuda else "cpu")
    print(device)

    # load target curves
    target_curve_root = os.path.join('data', 'testing_data_other_papers.joblib')
    print('testing root:', target_curve_root)
    data = load(target_curve_root)

    # load classifier model
    cls = classifier().double().cuda(args.gpu_id)
    optimizer = optim.Adam(cls.parameters(), lr=args.lr,weight_decay=1e-5)
    checkpoint_path = 'models/model_classifier/model_cls_best.pth'
    load_checkpoint(checkpoint_path, cls, optimizer)

    # organize path of models and data
    data_b_nums = [
    ('GCCC',7),('GCRR',6),('GRCR',9),('GRRC',8),
    ('RRR1',4),('RRR2',4),('RRR3',9),('RRR4',6),
    ]
    model_data_path = []
    for (inv, n_c) in data_b_nums:
        if 'G' in inv: t_num = 14560
        else: t_num = 12896
        model_dir = 'models/{}/model_{}_data_b_1215_{}_{}'.format(inv,inv,t_num,n_c)
        model_name = 'model_{}_data_b_1215_{}_best.pth'.format(inv,n_c)
        model_path = os.path.join(model_dir, model_name)

        data_dir = 'data/{}'.format(inv)
        data_name = 'data_b_1215_{}_{}.joblib'.format(t_num,n_c)
        data_path = os.path.join(data_dir, data_name)

        model_data_path.append([model_path, data_path])

    # path generation
    Ls = np.zeros((len(data),9))
    successful_cases = [24,29] # 2,3,6*,10*,14*',15*',16*',20
    compete = False
    for i, (fn, p1, p1_f, sp) in enumerate(data):
        if not(i in successful_cases):
            continue
        print('\n{}'.format(fn))
        sp_cls = torch.from_numpy(sp).cuda(args.gpu_id)

        # classification
        with torch.no_grad():
            cls.eval()
            criterion = nn.CrossEntropyLoss()
            output = cls(sp_cls.unsqueeze(0))
            _, inversion = torch.max(output, 1)
            args.select_dir = data_b_nums[inversion][0]
            print('Inversion: {}'.format(args.select_dir))

        # load model
        model = Net_2().double().cuda(args.gpu_id)
        optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=1e-5)
        checkpoint_path = model_data_path[inversion][0]
        load_checkpoint(checkpoint_path, model, optimizer)

        # load trainset to scale sp
        train_root = model_data_path[inversion][1]
        print('train root:', train_root)
        trainset = FDs(root=train_root, select_dir=args.select_dir)
        sp_scaled = trainset.scale.transform(np.expand_dims(sp, axis=0))
        sp_scaled = torch.from_numpy(sp_scaled).cuda(args.gpu_id)

        # predict results
        with torch.no_grad():
            model.eval()
            link = model(sp_scaled).cpu().numpy()

        # calcualate mechanism and plot the results
        p1_f = np.expand_dims(p1_f, axis=0)
        mde, L = matching3(link, p1_f, p1, fn, args, plot=1, compete=compete)
        print(mde)
        Ls[i,:] = L
    # np.savetxt('testing data/other papers/mechanisms.txt', Ls, fmt='%.5f', delimiter=',')


def main():
    # parse args
    args = parse_args()

    args.use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(args.gpu_id) if args.use_cuda else "cpu")
    print(device)

    best_b_model = True
    test_paper_cases = False

    if args.net == 'Net_1':
        model = Net_1().double().cuda(args.gpu_id)
    elif args.net == 'Net_2':
        model = Net_2().double().cuda(args.gpu_id)
    elif args.net == 'Net_3':
        model = Net_3().double().cuda(args.gpu_id)
    elif args.net == 'Net_4':
        model = Net_4().double().cuda(args.gpu_id)
    optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=1e-5)

    if 'G' in args.select_dir:
        train_num = 14560
    else:
        train_num = 12896
    test_num = 2496
    # torch.manual_seed(1)
    models = [
              # 'model_{}_data_e'.format(args.select_dir),
              # 'model_{}_data_1215'.format(args.select_dir),
              ]
    data = [
            # 'data_e_{}.joblib',
            # 'data_1215_{}.joblib',
            ]
    data_b_nums = {
    'GCCC':([7,5,6,4,11],8), # 0,8
    'GCRR':([6,7,2,11,4],0),  # 0
    'GRCR':([9,7,8,10,3],5),  # 0,3,5
    'GRRC':([8,6,3,10,4],1),  # 1,3,11
    'RRR1':([4,5,6,3,9],5),    # 4,7
    'RRR2':([4,5,6,2,3],8),
    'RRR3':([9,3,4,8,2],12),
    'RRR4':([6,4,5,3,7],9),
    }
    for i in data_b_nums[args.select_dir][0]:
        models.append('model_{}_data_b_1215_{}'.format(args.select_dir,i))
        data.append('data_b_1215_{}_'+str(i)+'.joblib')
        if best_b_model:
            break

    losses = np.zeros((len(models), len(data)))
    mdes = np.zeros((len(models), len(data)))
    for i in range(len(models)):
        # create data and model path
        if '_b_' in models[i]:
            model_dir = models[i][:23]+'{}_'.format(train_num)+models[i][23:]
            train_root = os.path.join(args.data_dir, models[i][11:23]+'{}_{}.joblib'.format(train_num,models[i][23:]))
            checkpoint_path = os.path.join(args.save_dir, model_dir, models[i]+'_best.pth')
        else:
            model_dir = models[i]+'_{}'.format(train_num)
            train_root = os.path.join(args.data_dir, models[i][11:]+'_{}.joblib'.format(train_num))
            checkpoint_path = os.path.join(args.save_dir, model_dir, models[i]+'_best.pth')

        print('train root:', train_root, '\n')
        trainset = FDs(root=train_root, select_dir=args.select_dir)

        load_checkpoint(checkpoint_path, model, optimizer)

        if test_paper_cases:
            # load testing data other papers
            valid_root = os.path.join('data', 'testing_data_other_papers.joblib')
            print('testing root:', valid_root)
            valset = FDs(root=valid_root, scale=trainset.scale, select_dir=args.select_dir, train=False)
            valset_loader = DataLoader(valset, batch_size=16, shuffle=False, num_workers=args.num_workers)

            test_mde = test(model, i, valset_loader, 0, args)
            continue

        if best_b_model:
            valid_root = os.path.join(args.data_dir, data[-1].format(test_num))
            print('testing root:', valid_root)
            valset = FDs(root=valid_root, scale=trainset.scale, select_dir=args.select_dir, valid=True)

            fn = [models[i], data[-1].split('.')[0].format(test_num)]
            n_clusters = data_b_nums[args.select_dir][0][0]
            begin = data_b_nums[args.select_dir][1]
            val_losses, val_mde = model_comparison(model, valset, fn, n_clusters, begin, args)
            continue

        for j in range(len(data)):
            valid_root = os.path.join(args.data_dir, data[j].format(test_num))
            print('testing root:', valid_root)
            valset = FDs(root=valid_root, scale=trainset.scale, select_dir=args.select_dir, valid=True)

            valset_loader = DataLoader(valset, batch_size=16, shuffle=False, num_workers=args.num_workers)
            val_losses, val_mde = valid(model, 2, valset_loader, args)
            losses[i,j] = val_losses.avg
            mdes[i,j] = val_mde.avg

    # np.savetxt('testing data/{}/clustering_experiment_loss_t5.txt'.format(args.select_dir), losses, fmt='%.5f', delimiter=',')
    # np.savetxt('testing data/{}/clustering_experiment_mde_t5.txt'.format(args.select_dir), mdes, fmt='%.5f', delimiter=',')


if __name__ == '__main__':
    # main()
    completed_model_test()
