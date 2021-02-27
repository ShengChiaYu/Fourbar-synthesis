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
np.random.seed(1216)
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from cmath import sqrt
from math import cos, sin, atan, acos, pi, degrees, radians

from dataset import image_data_3
from models import resnet50, resnet152
from utils import AverageMeter, parse_args


def select_test_image_2(image_fn, model, args):

    model.eval()
    root = join(os.getcwd(), 'test_images')

    dict = {'Ge_closed_curve.png':0, 'Ge_open_curve.png':1}
    test_images = pd.read_csv(join(root, 'test_image_mechanism.csv'))
    t_params = test_images.loc[dict[image_fn],:]

    image_fn = join(root, image_fn)
    print('\nTest image: ' + image_fn)
    image = cv2.imread(image_fn)
    image = cv2.resize(image,(200,200))
    (h, w, _) = image.shape
    center = (w / 2, h / 2)
    scale = 1
    r_params_list = []

    # M = cv2.getRotationMatrix2D(center, 45, scale)
    # rotated = cv2.warpAffine(image, M, (h, w), borderValue=0)
    # # 顯示圖片
    # cv2.imshow('My Image', rotated)
    #
    # # 按下任意鍵則關閉所有視窗
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # sys.exit()

    for angle in range(360):
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (h, w), borderValue=255)

        transform=transforms.Compose([
                  transforms.ToTensor(),
        ])
        rotated = transform(rotated).unsqueeze(0).cuda(args.gpu_id)
        r_param = model(rotated)

        r_params_list.append(r_param)

    r_params = torch.cat(r_params_list, dim=0)

    return t_params, r_params


def select_test_image_3(dataset, image_ind, model, args):
    num, image, label, t_params = dataset.__getitem__(image_ind)
    t_params = t_params.reshape(1,-1)
    r_params = model(image.cuda().unsqueeze(0))

    return t_params, r_params


def path_gen_open(L, th1, r, alpha, n, x0, y0):

    # Validity: Check if the linkages can be assembled and move.
    if max(L) >= (sum(L) - max(L)):
        # print("Impossible geometry.")
        return 0, 0

    # Limit of rotation angle of input linkage
    condition_1 = (L[0] + L[1]) - (L[2] + L[3])
    condition_2 = abs(L[0] - L[1]) - abs(L[2] - L[3])

    # Upper limit exists
    if condition_1 > 0  and condition_2 >= 0:
        th2_max = acos((L[0]**2 + L[1]**2 - (L[2] + L[3])**2) / (2*L[0]*L[1]))
        th2 = np.linspace(-th2_max, th2_max, n)
        # plt.title("Upper limit exists.")
        # plt.xlabel("Th_2 min = {:.3f}, Th_2 max = {:.3f}".format(degrees(-th2_max.real), degrees(th2_max.real)))
        # print("Upper limit exists.")
    # Lower limit exists
    elif condition_1 <= 0 and condition_2 < 0:
        th2_min = acos((L[0]**2 + L[1]**2 - (L[2] - L[3])**2) / (2*L[0]*L[1]))
        th2 = np.linspace(th2_min, 2*pi - th2_min, n)
        # plt.title("Lower limit exists.")
        # plt.xlabel("Th_2 min = {:.3f}, Th_2 max = {:.3f}".format(degrees(th2_min.real), degrees(2*pi-th2_min.real)))
        # print("Lower limit exists.")
    # Both limit exist
    elif condition_1 > 0 and condition_2 < 0:
        th2_max = acos((L[0]**2 + L[1]**2 - (L[2] + L[3])**2) / (2*L[0]*L[1]))
        th2_min = acos((L[0]**2 + L[1]**2 - (L[2] - L[3])**2) / (2*L[0]*L[1]))
        th2 = np.linspace(th2_min, th2_max, n)
        #th2 = np.linspace(-th2_max, -th2_min, n)
        # plt.title("Both limit exist.")
        # plt.xlabel("Th_2 min = {:.3f}, Th_2 max = {:.3f}".format(degrees(th2_min.real), degrees(th2_max.real)))
        # print("Both limit exist.")
    # No limit exists
    elif condition_1 <= 0 and condition_2 >= 0:
        th2 = np.linspace(0, 2*pi, n)
        # plt.title("No limit exists.")
        # plt.xlabel("Th_2 min = 0, Th_2 max = 360")
        # print("No limit exists.")

    # Calculate the positions of coupler curve by different input angles
    p1 = []
    p2 = []
    for i in range(n):
        k1 = L[0]**2 + L[1]**2 + L[2]**2 - L[3]**2 - 2*L[0]*L[1]*cos(th2[i]-th1)
        k2 = 2*L[0]*L[2]*cos(th1) - 2*L[1]*L[2]*cos(th2[i])
        k3 = 2*L[0]*L[2]*sin(th1) - 2*L[1]*L[2]*sin(th2[i])
        a = k1 + k2
        b = -2 * k3
        c = k1 -k2

        x_1 = (-b + sqrt(b**2 - 4*a*c).real) / (2 * a) # x_1 and x_2 = tan((1/2)*th3)
        x_2 = (-b - sqrt(b**2 - 4*a*c).real) / (2 * a)

        th3_1 = 2*atan(x_1)
        th3_2 = 2*atan(x_2)

        p1x = L[1]*cos(th2[i]) + r*cos(alpha+th3_1) + x0
        p1y = L[1]*sin(th2[i]) + r*sin(alpha+th3_1) + y0
        p1.append([p1x, p1y])

        p2x = L[1]*cos(th2[i]) + r*cos(alpha+th3_2) + x0
        p2y = L[1]*sin(th2[i]) + r*sin(alpha+th3_2) + y0
        p2.append([p2x, p2y])

        # plt.plot(L[1]*cos(th2[i]) + x0, L[1]*sin(th2[i]) + y0, 'go', markersize=1)

    p1 = np.array(p1)
    p2 = np.array(p2)

    return p1, p2


def path_plot_2(t_params, r_params, ind):
    n = 512
    t_points, _ = path_gen_open(t_params[:4], t_params[4], t_params[5], t_params[6], n, t_params[7], t_params[8])
    r_points, _ = path_gen_open(r_params[:4], r_params[4], r_params[5], r_params[6], n, r_params[7], r_params[8])

    # find the width and height of the test image and retrieved image
    t_width = np.max(t_points[:,0]) - np.min(t_points[:,0])
    t_height = np.max(t_points[:,1]) - np.min(t_points[:,1])
    r_width = np.max(r_points[:,0]) - np.min(r_points[:,0])
    r_height = np.max(r_points[:,1]) - np.min(r_points[:,1])

    # compare to find an appropriate scale method: width or height scaler
    w_scale = (t_width/r_width)
    h_scale = (t_height/r_height)
    width_err = np.abs(r_width*h_scale - t_width)
    height_err = np.abs(r_height*w_scale - t_height)
    if width_err < height_err:
        scale = h_scale
    else:
        scale = w_scale

    # rescale the retrieved image
    r_params_scaled = np.copy(r_params)
    r_params_scaled[:4] = r_params_scaled[:4] * scale
    r_params_scaled[5] = r_params_scaled[5] * scale

    # calculate the path with scaled mechanism
    r_points_scaled, _ = path_gen_open(r_params_scaled[:4], r_params_scaled[4], r_params_scaled[5],
                                       r_params_scaled[6], n, r_params_scaled[7], r_params_scaled[8])

    # find the center of images
    t_x = (np.max(t_points[:,0]) + np.min(t_points[:,0])) / 2
    t_y = (np.max(t_points[:,1]) + np.min(t_points[:,1])) / 2
    r_scaled_x = (np.max(r_points_scaled[:,0]) + np.min(r_points_scaled[:,0])) / 2
    r_scaled_y = (np.max(r_points_scaled[:,1]) + np.min(r_points_scaled[:,1])) / 2

    # move the x0 and y0 of retrieved mechanism
    r_params_scaled[7] = t_x - r_scaled_x
    r_params_scaled[8] = t_y - r_scaled_y

    # calcualte the final path and mse
    r_points_scaled, _ = path_gen_open(r_params_scaled[:4], r_params_scaled[4], r_params_scaled[5],
                                       r_params_scaled[6], n, r_params_scaled[7], r_params_scaled[8])
    # e = r_points_scaled - t_points
    # e = r_params_scaled - t_params
    # mse = np.mean(np.square(e))

    plt.title('Synthesis for Path Generation')
    r_points_wo_rot, _ = path_gen_open(r_params[:4], 0, r_params[5], r_params[6], n, r_params[7], r_params[8])
    plt.plot(t_points[:,0],t_points[:,1],'ro', markersize=1, label='Given path')
    plt.plot(r_points_wo_rot[:,0],r_points_wo_rot[:,1],'k-', markersize=1, label='Regressed image')
    plt.plot(r_points_scaled[:,0],r_points_scaled[:,1],'b-', markersize=1, label='Synthesized path')
    plt.plot(0,0,'k+')
    plt.axis('equal')
    plt.legend()
    save_path = os.path.join(os.getcwd(), 'test_images', 'image_regression',
    '{}.png'.format(ind))
    plt.savefig(save_path)
    plt.cla()

    return r_params_scaled


def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    # optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)


def image_regression(Inversion, dataset, model, args):
    # t_params, r_params = select_test_image_2('Ge_closed_curve.png', model, args)
    # t_params = t_params.to_numpy()
    # indices = np.arange(20)
    print(dataset.len)
    indices = np.random.randint(dataset.len, size=20)
    for ind in indices:
        t_params, r_params = select_test_image_3(dataset, ind, model, args)
        t_params = t_params.cpu()
        r_params = r_params.cpu()
        # print(r_params.shape)
        # print(t_params)
        # print(r_params)

        dirs = {'GCCC':0, 'GCRR':1,'GRCR':2,'GRRC':3}
        slices = [[1,2,3,5,6],[0,2,3,5,6],[0,1,3,5,6],[0,1,2,5,6]]
        select_dir = dirs[Inversion]
        slice = slices[select_dir]

        L = np.zeros((2,9))
        L[:,select_dir] = 1
        j = 0
        for i in slice:
            L[0,i] = np.copy(t_params[:,j])
            L[1,i] = np.copy(r_params[:,j])
            j += 1

        r_params_scaled = path_plot_2(L[0,:], L[1,:], ind)

        # print(r_params_scaled)

    #
    #
    # e1 = L[:,:4] - t_params[:4]/t_params[select_dir]
    # e2 = L[:,5] - t_params[5]/t_params[select_dir]
    # e = np.concatenate((e1,e2.reshape(-1,1)), axis=1)
    # mse = np.mean(np.square(e), axis=1)
    # ind = np.argsort(mse, axis=0) + 1
    # sorted_a = np.sort(mse, axis=0)
    # print(ind[:5])
    # print(L[351,:])
    # print(t_params[:4]/t_params[select_dir])
    # print(t_params[5]/t_params[select_dir])
    #
    # # save matching images with different rotation angles
    # all_mse = np.zeros((r_params.shape[0],1))
    # pbar = tqdm(total=r_params.shape[0], ncols=10, leave=True)
    # for i in range(L.shape[0]):
    #     angle = 360 - i
    #     L[i][4] = radians(angle)
    #     mse = path_plot_2(t_params, L[i], angle)
    #     all_mse[i,0] = mse
    #     pbar.update()
    # pbar.close()
    #
    # ind = np.argsort(all_mse, axis=0) + 1
    # sorted_a = np.sort(all_mse, axis=0)
    #
    # print(ind[:5,0])
    # print(sorted_a[:5,0])
    pass


def main():
    # parse args
    args = parse_args()

    args.use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(args.gpu_id) if args.use_cuda else "cpu")

    # create model
    checkpoint_path = join(os.getcwd(), 'models', 'model_GCRR_resnet50_best.pth')
    # model = resnet152(args).cuda(args.gpu_id)
    model = resnet50(args).cuda(args.gpu_id)
    load_checkpoint(checkpoint_path, model)

    # testing dataset
    model.eval()
    root = join(os.getcwd(), '../Encoder')
    transform=transforms.Compose([
              transforms.ToTensor(),
    ])
    dirs = {'GCCC':0, 'GCRR':1,'GRCR':2,'GRRC':3}
    select_dir = dirs[args.select_dir]
    dataset = image_data_3(root=join(root,'image_data_valid'), transform=transform, class_num=args.class_num, select_dir=select_dir)

    # image matching
    with torch.no_grad():
        image_regression(args.select_dir, dataset, model, args)


if __name__ == '__main__':
    main()
