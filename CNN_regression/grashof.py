import glob
import os
import sys
import argparse
from os.path import join

import numpy as np
np.random.seed(1216)
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from cmath import sqrt
from math import cos, sin, atan, acos, pi, degrees, floor, ceil
from itertools import permutations


def combvec(arr1, arr2):
    new_arr = []
    for i in arr1:
        for j in arr2:
            new_arr.append(np.append(i,j).tolist())
    return np.array(new_arr)


def path_gen_open(L, th1, r, alpha, n, x0, y0):

    # Validity: Check if the linkages can be assembled and move.
    if max(L) >= (sum(L) - max(L)):
        print("Impossible geometry.")
        return 0, 0

    # Limit of rotation angle of input linkage
    condition_1 = (L[0] + L[1]) - (L[2] + L[3])
    condition_2 = abs(L[0] - L[1]) - abs(L[2] - L[3])

    # Upper limit exists
    if condition_1 > 0  and condition_2 >= 0:
        th2_max = acos((L[0]**2 + L[1]**2 - (L[2] + L[3])**2) / (2*L[0]*L[1]))
        th2 = np.linspace(-th2_max, th2_max, n)
        # plt.title("Upper limit exists.")
        plt.xlabel("Th_2 min = {:.3f}, Th_2 max = {:.3f}".format(degrees(-th2_max.real), degrees(th2_max.real)))
        # print("Upper limit exists.")
    # Lower limit exists
    elif condition_1 <= 0 and condition_2 < 0:
        th2_min = acos((L[0]**2 + L[1]**2 - (L[2] - L[3])**2) / (2*L[0]*L[1]))
        th2 = np.linspace(th2_min, 2*pi - th2_min, n)
        # plt.title("Lower limit exists.")
        plt.xlabel("Th_2 min = {:.3f}, Th_2 max = {:.3f}".format(degrees(th2_min.real), degrees(2*pi-th2_min.real)))
        # print("Lower limit exists.")
    # Both limit exist
    elif condition_1 > 0 and condition_2 < 0:
        th2_max = acos((L[0]**2 + L[1]**2 - (L[2] + L[3])**2) / (2*L[0]*L[1]))
        th2_min = acos((L[0]**2 + L[1]**2 - (L[2] - L[3])**2) / (2*L[0]*L[1]))
        th2 = np.linspace(th2_min, th2_max, n)
        #th2 = np.linspace(-th2_max, -th2_min, n)
        # plt.title("Both limit exist.")
        plt.xlabel("Th_2 min = {:.3f}, Th_2 max = {:.3f}".format(degrees(th2_min.real), degrees(th2_max.real)))
        # print("Both limit exist.")
    # No limit exists
    elif condition_1 <= 0 and condition_2 >= 0:
        th2 = np.linspace(0, 2*pi, n)
        # plt.title("No limit exists.")
        plt.xlabel("Th_2 min = 0, Th_2 max = 360")
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
        th3_1 = 2*atan(x_1)
        p1x = L[1]*cos(th2[i]) + r*cos(alpha+th3_1) + x0
        p1y = L[1]*sin(th2[i]) + r*sin(alpha+th3_1) + y0
        p1.append([p1x, p1y])

        # x_2 = (-b - sqrt(b**2 - 4*a*c).real) / (2 * a)
        # th3_2 = 2*atan(x_2)
        # p2x = L[1]*cos(th2[i]) + r*cos(alpha+th3_2) + x0
        # p2y = L[1]*sin(th2[i]) + r*sin(alpha+th3_2) + y0
        # p2.append([p2x, p2y])

        # plt.plot(L[1]*cos(th2[i]) + x0, L[1]*sin(th2[i]) + y0, 'go', markersize=1)

    p1 = np.array(p1)
    # p2 = np.array(p2)

    return p1


def random_params(set_num, link):

    max_link_ratio = 5
    s = np.full((set_num,1), 1)
    p = 1 + (max_link_ratio-1)*np.random.rand(set_num,1)
    l = p + (5-p)*np.random.rand(set_num,1)
    q = s+l-p + (l-(s+l-p))*np.random.rand(set_num,1)
    r6 = 1 + (5-1)*np.random.rand(set_num,1)
    theta6 = 2*pi*np.random.rand(set_num,1)

    L = np.zeros((set_num,9))
    L[:,link[0]] = s[:,0]
    L[:,link[1]] = l[:,0]
    L[:,link[2]] = p[:,0]
    L[:,link[3]] = q[:,0]
    L[:,5] = r6[:,0]
    L[:,6] = theta6[:,0]

    return L


def log_random_params(set_num, link):

    def log_normal_rand(mu, sigma, set_num):
        tmp = np.random.lognormal(mu, sigma, (set_num,1))
        tmp = tmp / np.max(tmp)
        # print(tmp)
        return tmp

    max_link_ratio = 5
    mu, sigma = 0., 0.6 # mean and standard deviation
    s = np.full((set_num,1), 1)
    p = 1 + (max_link_ratio-1)*log_normal_rand(mu, sigma, set_num)
    l = p + (max_link_ratio-p)*log_normal_rand(mu, sigma, set_num)
    q = s+l-p + (l-(s+l-p))*log_normal_rand(mu, sigma, set_num)

    r6 = 1 + (max_link_ratio-1)*np.random.rand(set_num,1)
    theta6 = 2*pi*np.random.rand(set_num,1)
    # count, bins, ignored = plt.hist(q, 100, align='mid')
    # plt.show()
    L = np.zeros((set_num,9))
    L[:,link[0]] = s[:,0]
    L[:,link[1]] = l[:,0]
    L[:,link[2]] = p[:,0]
    L[:,link[3]] = q[:,0]
    L[:,5] = r6[:,0]
    L[:,6] = theta6[:,0]

    return L


def exhausted_params(n, link):
    # n: precision
    # p-link and chosen 1 < p < 5 at random.
    # l-link and chosen p < l <= 5 at random.
    # q-link and chosen s+l-p < q < l at random.

    fourbar = [];
    max_link_ratio = 5
    s = 1;
    for p in np.arange(s+n,max_link_ratio,n):
        for l in np.arange(p+n,max_link_ratio,n):
            for q in np.arange(s+l-p+n,l,n):
                fourbar.append([s,l,p,q])
    fourbar = np.array(fourbar)
    print('Combination number of fourbar: {}'.format(fourbar.shape))

    # r6 is chosen 1 <= r6 <= 5 at random.
    # theta6 is chosen 0 <= theta6 <= 2*pi at random.
    r6 = np.arange(1,5,0.25)
    fourbar = combvec(fourbar, r6);
    print('Combination number of fourbar&r6: {}'.format(fourbar.shape))

    theta6 = np.arange(0,2*pi,0.25)
    fourbar = combvec(fourbar, theta6)
    print('Combination number of fourbar&r6&theta6: {}'.format(fourbar.shape))

    L = np.zeros((fourbar.shape[0],9))
    L[:,link[0]] = fourbar[:,0]
    L[:,link[1]] = fourbar[:,1]
    L[:,link[2]] = fourbar[:,2]
    L[:,link[3]] = fourbar[:,3]
    L[:,5] = fourbar[:,4]
    L[:,6] = fourbar[:,5]

    return L


def image_generator(root, n, Inversion, link, start_ind, N):
    # Combination for training data
    # L = exhausted_params(n, link)
    # L = log_random_params(15000, link)

    # Combination for testing data
    L = random_params(5000, link)

    # Output mechanism dimensions
    csv_save_path = join(root, Inversion)
    part_div_num = 1000
    dir_num = ceil(start_ind/part_div_num)
    image_save_path = join(csv_save_path, str(link[1]+1), 'part_{:04d}'.format(dir_num))

    mechanism_dim = open(join(csv_save_path, '{}_label.csv'.format(link[1]+1)), 'w')
    generator_pbar = tqdm(total=20000-start_ind+1, ncols=10, leave=True)

    for i in range(start_ind-1,20000):
        j = i - 14999
        mechanism_dim.write('{},{},{},{},{},{},{},{},{},\n'.format(L[j,0], L[j,1], L[j,2],
                            L[j,3], L[j,4], L[j,5], L[j,6], L[j,7], L[j,8]))
        mechanism_dim.flush()
        p1 = path_gen_open(L[j,:4], L[j,4], L[j,5], L[j,6], N, L[j,7], L[j,8])
        plt.plot(p1[:,0],p1[:,1],'ko')
        plt.axis('equal')
        plt.axis('off')

        if i%part_div_num == 0:
            if i != 0:
                dir_num += 1
            try:
                image_save_path = join(csv_save_path, str(link[1]+1), 'part_{:04d}'.format(dir_num))
                os.makedirs(image_save_path)
            except FileExistsError:
                print("Directory ", image_save_path, " already exists")


        plt.savefig(join(image_save_path, '{:06d}.png'.format(i+1)))
        plt.cla()

        generator_pbar.update()

    generator_pbar.close()


def grashof(root, n, Inversion, set, start_ind):
    # Permutation of Grashof four-bar mechanism
    link_permutations = list(permutations([0,1,2,3]))
    set_names = ['GCCC','GCRR','GRCR','GRRC']
    link_set = {}
    for i in range(len(set_names)):
        link_set[set_names[i]] = link_permutations[i*6:(i+1)*6]

    # Choose links of inversion and set
    N = 512;
    links = link_set[Inversion]
    link = links[2*set-1]
    print('{}, set {}, link-s/l: {}/{}'.format(Inversion, set, link[0]+1, link[1]+1))
    print('Start index: {}'.format(start_ind))

    try:
        image_save_path = join(root, Inversion, str(link[1]+1))
        os.makedirs(image_save_path)
    except FileExistsError:
        print("Directory ", image_save_path, " already exists")
    finally:
        image_generator(root, n, Inversion, link, start_ind, N);


def main():
    root = join(os.getcwd(), 'image_data_logn-d')
    precision = 0.25
    inversion = 'GCRR'
    set = 1
    start_ind = 15000

    grashof(root=root, n=precision, Inversion=inversion, set=set, start_ind=start_ind)


if __name__ == '__main__':
    main()
