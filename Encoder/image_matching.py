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
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from cmath import sqrt
from math import cos, sin, atan, acos, pi, degrees, radians

from dataset import image_data_3
from models import autoencoder
from utils import AverageMeter, parse_args


def read_features(file_name):
    file_path = join(os.getcwd(), 'features', file_name)
    my_data = np.genfromtxt(file_path, delimiter=',')
    image_index = torch.from_numpy(my_data[:,0]).float()
    features = torch.from_numpy(my_data[:,1:-1]).float()
    print('\nLoad {} features...'.format(features.shape[0]))

    return image_index, features


def read_mechanism_parameters(dir_name):
    dir = join(os.getcwd(), 'image_data_3', dir_name)
    label_csvs = sorted(glob.glob(join(dir, '*.csv')))
    print('\nLoad mechanism parameters: \n{}'.format(dir))
    data = []
    for i, label_csv in enumerate(label_csvs):
        print(label_csv)
        params = np.genfromtxt(label_csv, delimiter=',')
        params = torch.from_numpy(params[:,:-1]).float()
        data.append(params)
    data = torch.cat(data, dim=0)

    print('Finish loading {} sets of parameters'.format(len(data)))

    return data


def select_test_image_2(model, args):
    model.eval()
    root = join(os.getcwd(), 'test_images')

    test_images = pd.read_csv(join(root, 'test_image_mechanism.csv'))
    t_params = test_images.loc[0,:]

    image_fn = join(root, 'Ge_closed_curve.png')
    print('\nTest image: ' + image_fn)
    image = cv2.imread(image_fn, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image,(200,200))
    (h, w) = image.shape
    center = (w / 2, h / 2)
    scale = 1
    test_feature_list = []

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
        test_feature = model(rotated, feature=True)

        test_feature_list.append(test_feature)

    test_features = torch.cat(test_feature_list, dim=0)

    return t_params, test_features


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


def path_plot_2(t_params, r_params, rank, retrieved_ind, se, angle):
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
    r_params_scaled = r_params.clone()
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

    # calcualte the final path
    r_points_scaled, _ = path_gen_open(r_params_scaled[:4], r_params_scaled[4], r_params_scaled[5],
                                       r_params_scaled[6], n, r_params_scaled[7], r_params_scaled[8])

    plt.title('Synthesis for Path Generation')
    r_points_wo_rot, _ = path_gen_open(r_params[:4], 0, r_params[5], r_params[6], n, r_params[7], r_params[8])
    plt.plot(t_points[:,0],t_points[:,1],'ro', markersize=1, label='Given path')
    plt.plot(r_points_wo_rot[:,0],r_points_wo_rot[:,1],'k-', markersize=1, label='Retrieved image')
    plt.plot(r_points_scaled[:,0],r_points_scaled[:,1],'b-', markersize=1, label='Synthesized path')
    plt.plot(0,0,'k+')
    plt.axis('equal')
    plt.legend()
    save_path = os.path.join(os.getcwd(), 'test_images', 'image_matching',
    '{}_{}_{:.2E}_{}.png'.format(rank, retrieved_ind, se, angle))
    plt.savefig(save_path)
    plt.cla()


def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    # optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)


def image_matching(feature_filename, dir_name, model, args):
    image_index, features = read_features(feature_filename)
    params = read_mechanism_parameters(dir_name)
    t_params, test_features = select_test_image_2(model, args)

    # t_params = torch.Tensor(params[0])
    # test_features = features[0].reshape(1,-1)
    # print(test_features.shape)
    # print(test_features)

    features, test_features = features.cuda(), test_features.cuda()

    all_angle_list = []
    retrieved_img = []
    for i in range(test_features.shape[0]):
        test_feature = test_features[i].repeat(features.shape[0],1)
        err = (features-test_feature)
        se = (err*err).sum(dim=1, keepdim=True)
        sorted_se, indices = torch.sort(se,0)
        if indices[0].item() not in retrieved_img:
            retrieved_img.append(indices[0].item())
            tmp = torch.Tensor([sorted_se[0], indices[0]])
            all_angle_list.append(tmp)


    all_angle = torch.cat(all_angle_list, dim=0).reshape(-1,2)
    sorted_se, indices = torch.sort(all_angle[:,0],0)

    # save matching images with different rotation angles
    for i in range(all_angle.shape[0]):
        retrieved_ind = int(all_angle[indices[i]][1].item())
        angle = 360 - indices[i].item()
        set = int(retrieved_ind / (features.shape[0]/3))+1

        # print('\n---Matching Result---')
        # print('Set of the retrieved image: {}'.format(set))
        # print('Number of the retrieved image: {}'.format(image_index[retrieved_ind]))
        # print('Minimum square error: {:.2E}'.format(sorted_se[i].item()))
        # print('Theta 1 = {}'.format(angle))

        r_params = torch.Tensor(params[retrieved_ind])
        r_params[4] = radians(angle)
        path_plot_2(t_params, r_params, i+1, retrieved_ind, sorted_se[i], indices[i])


def main():
    # parse args
    args = parse_args()

    args.use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(args.gpu_id) if args.use_cuda else "cpu")

    # create model
    checkpoint_path = join(os.getcwd(), 'models', 'v3', 'model_GCRR_autoencoder.pth')
    # model = resnet152(pretrained=True).cuda(args.gpu_id)
    model = autoencoder().cuda(args.gpu_id)
    load_checkpoint(checkpoint_path, model)

    # select features and their parameters of training images
    # feature_filename = 'features_GCRR_resnet152.txt'
    feature_filename = 'features_GCRR_autoencoder.txt'
    dir_name = args.select_dir

    # image matching
    with torch.no_grad():
        image_matching(feature_filename, dir_name, model, args)


if __name__ == '__main__':
    main()
