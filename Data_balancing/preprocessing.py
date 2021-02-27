import glob
import os
import sys
import argparse
import pandas as pd
from os.path import join

import numpy as np
from numpy import linalg as LA
from numpy.fft import fft, fftfreq
from sklearn.decomposition import PCA
from joblib import dump, load
import matplotlib.pyplot as plt
import spatial_efd
import pywt
import scipy
from tqdm import tqdm

from PyQt5.QtGui import QPolygonF
from PyQt5.QtCore import QPointF
from math import cos, sin, atan, atan2, acos, pi, degrees, radians, floor, ceil
from cmath import exp
from itertools import permutations

from grashof import path_gen_open, path_gen_open_plot_links, exhaustive_params, random_params


def image_generator(grashof, inversion, n, link, N):
    open_curve_sets = ['GRCR','GRRC','RRR1','RRR2','RRR3','RRR4']
    if n < 3:
        # Combination for training data
        L = exhaustive_params(grashof, n, link)
        print('generate exhaustive dataset')
    else:
        # Combination for testing data
        L = random_params(grashof, n, link)
        print('generate random dataset')

    # Output mechanism dimensions
    generator_pbar = tqdm(total=L.shape[0], ncols=10, leave=True)
    p1s = []
    sps = []
    for i in range(L.shape[0]):
        p1 = path_gen_open(L[i,:4], L[i,4], L[i,5], L[i,6], 30, L[i,7], L[i,8])

        if inversion in open_curve_sets:
            p1 = bezier_curve(p1[:,0], p1[:,1], fn='', plot=0, nTimes=N)
        else:
            p1 = efd_fitting(p1[:,0],p1[:,1],fn='',plot=0, N=N)

        p1_normal = curve_normalization_pca2(p1, plot=0, trans=0, rot=0, scal=1)
        sp = plot_fft_plus_power(p1_normal, plot=0)

        p1s.append(p1)
        sps.append(sp)

        generator_pbar.update()

    generator_pbar.close()

    return np.array(L), np.array(p1s), np.array(sps)


def fourbar_mechanism(inversion, set, n=5, N=120):
    # Permutation of Grashof four-bar mechanism
    link_permutations = list(permutations([0,1,2,3]))
    grashof_set_names = ['GCCC','GCRR','GRCR','GRRC']
    non_grashof_set_names = ['RRR1','RRR2','RRR3','RRR4']
    link_set = {}
    for i in range(len(grashof_set_names)):
        link_set[grashof_set_names[i]] = link_permutations[i*6:(i+1)*6]
        link_set[non_grashof_set_names[i]] = link_permutations[i*6:(i+1)*6]

    # Choose links of inversion and set
    links = link_set[inversion]
    link = links[2*set-1]
    if inversion in grashof_set_names:
        print('{}, set {}, link-s/l: {}/{}'.format(inversion, set, link[0]+1, link[1]+1))
        grashof = True
    elif inversion in non_grashof_set_names:
        print('{}, set {}, link-l/s: {}/{}'.format(inversion, set, link[0]+1, link[1]+1))
        grashof = False

    return image_generator(grashof, inversion, n, link, N)


def is_open_curve(p1):
    # mean distances of all points
    dist = p1[:-1,:]-p1[1:,:]
    dist = LA.norm(dist,axis=1)
    mean_dist = np.mean(dist)

    # distance of the first and the last point
    dist_fnl = LA.norm(p1[0,:]-p1[-1,:])
    if dist_fnl > 3*mean_dist:
        return True

    return False


def efd_fitting(x,y,fn='',plot=0, N=40):
    nyquist = spatial_efd.Nyquist(x)
    tmpcoeffs = spatial_efd.CalculateEFD(x, y, nyquist)
    harmonic = spatial_efd.FourierPower(tmpcoeffs, x)
    coeffs = spatial_efd.CalculateEFD(x, y, harmonic)

    locus = spatial_efd.calculate_dc_coefficients(x, y)
    xt, yt = spatial_efd.inverse_transform(coeffs, harmonic=harmonic, locus=locus, n=N)
    p1 = np.vstack((xt, yt)).T

    if plot:
        fig, ax = plt.subplots(1,3,figsize=(12,6))
        ax[0].scatter(x,y,c='r',marker='.',label='raw_path')
        ax[1].scatter(xt,yt,c='b',marker='.',label='efd_path')
        x = np.hstack([x,x[0]])
        y = np.hstack([y,y[0]])
        ax[2].plot(x,y,'r',label='raw_path')
        ax[2].scatter(xt,yt,c='b',marker='.',label='efd_path')
        ax[0].axis('equal')
        ax[1].axis('equal')
        ax[2].axis('equal')
        # ax[0].set_title('(a)')
        # ax[1].set_title('(b)')
        # ax[2].set_title('(c)')

        plt.savefig('./test_figure/fitting/{}_efd.png'.format(fn),bbox_inches='tight')

    return p1


def bezier_curve(x, y, fn='', plot=0, nTimes=120):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       points should be a list of lists, or list of tuples
       such as [ [1,1],
                 [2,3],
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """
    def bernstein_poly(i, n, t):
        """
         The Bernstein polynomial of n, i as a function of t
        """
        return scipy.special.comb(n, i) * ( t**(n-i) ) * (1 - t)**i

    nPoints = x.shape[0]
    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])

    xt = np.dot(x, polynomial_array)
    yt = np.dot(y, polynomial_array)
    p1 = np.vstack((xt, yt)).T


    if plot:
        fig, ax = plt.subplots(1,3,figsize=(12,6))
        ax[0].scatter(x,y,c='r',marker='.',label='raw_path')
        ax[1].scatter(xt,yt,c='b',marker='.',label='bezier_path')
        ax[2].plot(x,y,'r',label='raw_path')
        ax[2].scatter(xt,yt,c='b',marker='.',label='bezier_path')
        # ax[0].legend(loc='upper left')
        # ax[1].legend(loc='upper left')
        # ax[2].legend(loc='upper left')
        ax[0].axis('equal')
        ax[1].axis('equal')
        ax[2].axis('equal')
        # ax[0].set_title('(a)')
        # ax[1].set_title('(b)')
        # ax[2].set_title('(c)')
        plt.savefig('./test_figure/fitting/{}_bezier.png'.format(fn),bbox_inches='tight')

    return p1


def calculate_rotation_angle(p1,fn=''):

    def axes_orientation(v1,v2,p1,fn='',plot=False):
        a = v1[1] / v1[0]
        b = y_mean - a*x_mean
        val = p1[:,1] - a*p1[:,0] - b
        neg_dist = LA.norm((p1[val<0]-np.mean(p1,axis=0)),axis=1).sum()
        pos_dist = LA.norm((p1[val>0]-np.mean(p1,axis=0)),axis=1).sum()
        # print(neg_dist)
        # print(pos_dist)
        v2 = np.abs(v2)
        if a < 0:
            if neg_dist > pos_dist:
                axis = [-v2[0],-v2[1]]
            else:
                axis = [v2[0],v2[1]]
        else:
            if neg_dist > pos_dist:
                axis = [v2[0],-v2[1]]
            else:
                axis = [-v2[0],v2[1]]
        if plot:
            pca = np.concatenate([v1,-v1,(0,0),axis]).reshape(4,2) + np.mean(p1, axis=0)
            fig, ax = plt.subplots(figsize=(8,4))
            ax.plot(p1[val>0,0],p1[val>0,1],'b^',label=r'$C_{pos}$'+' point set')
            ax.plot(p1[val<0,0],p1[val<0,1],'r.',label=r'$C_{neg}$'+' point set')
            ax.plot(pca[:2,0], pca[:2,1], 'k-',linewidth=2,label=r'$2^{nd}$'+' Pca')
            ax.plot(pca[2:,0], pca[2:,1], 'g--',linewidth=2,label='Positive direction of '+r'$1^{st}$'+' Pca')
            ax.legend(loc='upper left')
            plt.axis('equal')
            plt.savefig('./test_figure/{} {:.3f}.png'.format(fn, v2[1]/v2[0]),bbox_inches='tight')

        return axis

    ## Step 1: calculate the mean and std of path points
    x_mean, y_mean = np.mean(p1, axis=0)

    ## Step 2: calculate the prinpal component axes (eigenvectors)
    m = p1.shape[0]
    Cxx = np.sum((p1[:,0]-x_mean)**2) / m
    Cyy = np.sum((p1[:,1]-y_mean)**2) / m
    Cxy = np.sum((p1[:,0]-x_mean)*(p1[:,1]-y_mean)) / m
    C = np.array(((Cxx,Cxy),(Cxy,Cyy)))
    w, v = LA.eig(C)
    # print(w)
    # print(v)

    ## Step 3: calculate the orientation of the axes
    axes = []
    axes.append(axes_orientation(v[:,1],v[:,0],p1,fn,plot=0))
    axes.append(axes_orientation(v[:,0],v[:,1],p1,fn,plot=0))
    axes = np.array(axes)
    # print(axes)

    theta_1 = atan2(axes[0][1],axes[0][0])
    theta_2 = atan2(axes[1][1],axes[1][0])
    pca = np.insert(axes, 1, (0,0), axis=0) + np.mean(p1, axis=0)
    # print(degrees(theta_1))
    # print(degrees(theta_2))

    ## Step 4: calculate the rotation matrix
    if theta_1*theta_2 > 0:
        alpha = theta_1 if theta_1 < theta_2 else theta_2
    elif theta_1*theta_2 < 0:
        if np.abs(theta_1) < pi/2:
            alpha = theta_1 if theta_1 < theta_2 else theta_2
        else:
            alpha = theta_1 if theta_1 > theta_2 else theta_2
    else:
        alpha = -pi/2 if theta_1 == -pi/2 or theta_2 == -pi/2 else 0

    return alpha, pca


def curve_normalization_pca2(p1, fn='', plot=False, trans=0, rot=0, scal=1):
    #----------------------- INVARIANCE TESTING -------------------------------
    ## testing translation
    if trans:
        fn, p1 = "translation", p1 + np.array(trans)

    ## testing rotation
    if rot:
        fn, p1 = rotate(p1, theta=rot, center="origin")

    ## testing scale
    if scal != 1:
        fn, p1 = scale(p1, scale_ratio=scal)

    alpha, pca = calculate_rotation_angle(p1,fn)
    c, s = np.cos(alpha), np.sin(alpha)
    R = np.array(((c, s), (-s, c)))

    ## Step 5: normalized the path points
    x_mean, y_mean = np.mean(p1, axis=0)
    p1_normal = (p1 - np.array([x_mean, y_mean]))
    p1_normal = np.matmul(R,p1_normal.T).T
    Px = np.min(p1_normal[:,0])
    Py = np.min(p1_normal[:,1])
    Qx = np.max(p1_normal[:,0])
    Qy = np.max(p1_normal[:,1])
    w = Qx - Px
    p1_normal = p1_normal / w

    ## Step 6: find the starting point of the path
    r = LA.norm(p1_normal, axis=1)
    ind = np.argmin(r)
    str_p = p1_normal[ind,:]
    p1_normal_roll = np.roll(p1_normal,p1_normal.shape[0]-ind,axis=0)


    if plot:
        fig, axes = plt.subplots(1,2,figsize=(8,4))

        axes[0].plot(p1[:,0],p1[:,1],'b.',label='Initial')
        axes[0].plot(pca[:,0], pca[:,1], color='green', linestyle='dashed',linewidth=2,label='Pca')
        axes[0].legend(loc='upper left')
        axes[0].axis('equal')
        # axes[0].set_title('{} {:.3f}'.format(fn, degrees(alpha)))
        # ax[0].set_title('(a)')


        axes[1].plot(p1_normal[:,0],p1_normal[:,1],'.',color='orange',label='Normalized')
        pca = pca - np.array([x_mean, y_mean])
        pca = np.matmul(R,pca.T).T
        axes[1].plot(pca[:,0], pca[:,1], color='green', linestyle='dashed',linewidth=2,label='Pca')
        axes[1].plot(str_p[0],str_p[1],'r+',label='Start point')
        # axes[1].plot(p1_normal[0,0],p1_normal[0,1],'b+',label='1st p.')
        # axes[1].plot(p1_normal[10,0],p1_normal[10,1],'g+',label='10th p.')
        axes[1].legend(loc='upper left')
        axes[1].axis('equal')
        # ax[1].set_title('(b)')

        plt.savefig('./test_figure/curve_normalization_pca/{} {:.3f}.png'.format(fn, degrees(alpha)),bbox_inches='tight')

    return p1_normal_roll


def rotate(p1, theta=0, center="origin"):
    fn = "rotation"
    theta = radians(theta)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))

    if center == "mean":
        p1c = p1 - np.mean(p1, axis=0)
        p1 = np.matmul(R,p1c.T).T + np.mean(p1, axis=0)
    else:
        p1 = np.matmul(R,p1.T).T

    return fn, p1


def scale(p1, scale_ratio=1):
    fn = "Scaled_{}x".format(scale_ratio)
    Px = np.min(p1[:,0])
    Py = np.min(p1[:,1])
    Qx = np.max(p1[:,0])
    Qy = np.max(p1[:,1])
    w = Qx - Px
    h = Qy - Py

    p1 = (p1 - np.array((Px,Py)))*scale_ratio + np.array((Px,Py))

    return fn, p1


def multi_normalization(p1, p1_init_normal, times=5, plot=0):

    p1_normal = p1_init_normal
    p1_normals = []
    for i in range(times):
        p1_normal, str_p = curve_normalization(p1_normal, fn="", plot=0)
        p1_normals.append(p1_normal)
        print("{} normalized, Error: {:.3f}".format(i+2,LA.norm(p1_init_normal-p1_normal)))

    if plot:
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(p1[:,0],p1[:,1],label='original')
        ax.plot(p1_init_normal[:,0],p1_init_normal[:,1],label='normalized')
        for i in range(len(p1_normals)):
            ax.plot(p1_normals[i][:,0],p1_normals[i][:,1],label='{}'.format(i+1))
        # ax.legend(loc='upper left')
        plt.axis('equal')
        plt.savefig('./test_figure/curve_normalization/multi_normalization_{}.png'.format(times))


def plot_fft_plus_power(signal, fn='', plot=0):

    signal_complex = signal[:,0]+signal[:,1]*1j
    sp = fft(signal_complex)

    if plot:
        fig, axes = plt.subplots(1,2,figsize=(8,4))
        axes[0].plot(signal[:,0],'-',label='x')
        axes[0].plot(signal[:,1],'--',label='y')
        axes[0].legend(loc='upper left')
        # axes[0].set_title('{}_x+yi'.format(fn))
        # ax[0].set_title('(a)')
        axes[1].plot(sp.real,'-',label='Real.')
        axes[1].plot(sp.imag,'--',label='Imag.')
        axes[1].legend(loc='upper left')
        # axes[1].set_title('{}_fft_x+yi'.format(fn))
        # ax[1].set_title('(b)')

        plt.savefig('./test_figure/curve_normalization_pca/{}_fft.png'.format(fn),bbox_inches='tight')

    return np.hstack((sp.real, sp.imag))


def plot_wavelet(time, signal, scales, waveletname = 'cmor1.5-1.0'):
    cmap = plt.cm.seismic
    title = 'Wavelet Transform (Power Spectrum) of signal',
    ylabel = 'Period (years)',
    xlabel = 'Time'

    dt = time[1] - time[0]
    [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
    power = (abs(coefficients)) ** 2
    period = 1. / frequencies
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]
    contourlevels = np.log2(levels)

    fig, ax = plt.subplots(figsize=(15, 10))
    im = ax.contourf(time, np.log2(period), np.log2(power), contourlevels, extend='both',cmap=cmap)

    ax.set_title(title, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=18)

    yticks = 2**np.arange(np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max())))
    ax.set_yticks(np.log2(yticks))
    ax.set_yticklabels(yticks)
    ax.invert_yaxis()
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], -1)

    cbar_ax = fig.add_axes([0.95, 0.5, 0.03, 0.25])
    fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    plt.show()


def PolyArea(result):
    x = [p.x() for p in result]
    y = [p.y() for p in result]
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


def IOU(p1, p2):

    Qp1 = []
    Qp2 = []
    for i in range(p1.shape[0]):
        Qp1.append(QPointF(p1[i,0],p1[i,1]))
        Qp2.append(QPointF(p2[i,0],p2[i,1]))

    QPolygon1 = QPolygonF(Qp1)
    QPolygon2 = QPolygonF(Qp2)

    intersection = QPolygon1.intersected(QPolygon2)
    union = QPolygon1.united(QPolygon2)

    inter_area = PolyArea(intersection)
    union_area = PolyArea(union)

    return inter_area/union_area, intersection, union


def point_wise_mde(p1, p2, alpha):
    # point wise mean distances error
    # rotate
    c, s = np.cos(alpha), np.sin(alpha)
    R = np.array(((c, s), (-s, c)))
    p1 = p1 - np.mean(p1, axis=0)
    p1 = np.matmul(R,p1.T).T

    # adjust ratio depending on the diagonal
    P1x = np.min(p1[:,0])
    P1y = np.min(p1[:,1])
    Q1x = np.max(p1[:,0])
    Q1y = np.max(p1[:,1])

    P2x = np.min(p2[:,0])
    P2y = np.min(p2[:,1])
    Q2x = np.max(p2[:,0])
    Q2y = np.max(p2[:,1])

    p1_dig = np.sqrt((Q1x-P1x)**2+(Q1y-P1y)**2)
    p2_dig = np.sqrt((Q2x-P2x)**2+(Q2y-P2y)**2)
    ratio = p2_dig / p1_dig
    p1 = p1 * ratio

    # find the start point of p1
    r = LA.norm(p1, axis=1)
    ind = np.argmin(r)
    p1 = np.roll(p1,p1.shape[0]-ind,axis=0)

    # move to align p2
    p1 = p1 + np.mean(p2, axis=0)

    # find the start point of p2
    r = LA.norm(p2-np.mean(p2, axis=0), axis=1)
    ind = np.argmin(r)
    p2 = np.roll(p2,p2.shape[0]-ind,axis=0)

    # calculate distance error
    distance_error = np.sqrt((p1[:,0]-p2[:,0])**2+(p1[:,1]-p2[:,1])**2)

    # normalize error
    mde = np.mean(distance_error) / p2_dig

    return mde, p1, p2, ratio


def matching(ep, batch_idx, links, targets, fns, args, plot=False):

    dirs = {'GCCC':[[0],[1,2,3,5,6]],
            'GCRR':[[1],[0,2,3,5,6]],
            'GRCR':[[2],[0,1,3,5,6]],
            'GRRC':[[3],[0,1,2,5,6]],
            'RRR1':[[1],[0,2,3,5,6]],
            'RRR2':[[0],[1,2,3,5,6]],
            'RRR3':[[0],[2,1,3,5,6]],
            'RRR4':[[0],[3,1,2,5,6]]}
    slice = dirs[args.select_dir]
    n = links.shape[0]
    mde = np.ones(n)
    if plot:
        fig, axes = plt.subplots(1,n,figsize=(2*n,4))

    for i in range(n):
        link = links[i]
        target = targets[i]

        L = np.zeros(9)
        L[slice[0][0]] = 1
        for j in range(len(slice[1])):
            L[slice[1][j]] = link[j]

        predict = path_gen_open(L[:4], L[4], L[5], L[6], target.shape[0], L[7], L[8])
        if predict.shape[0] == 1:
            continue

        # adjust the predicted path to target path
        if args.select_dir in ['GCCC','GCRR']:
            predict = efd_fitting(predict[:,0],predict[:,1],plot=0,N=target.shape[0])
        else:
            predict = bezier_curve(predict[:,0],predict[:,1],plot=0,nTimes=target.shape[0])

        p_alpha, p_pca = calculate_rotation_angle(predict)
        t_alpha, t_pca = calculate_rotation_angle(target)

        tmp_mde = np.zeros(4)
        for j in range(4):
            alpha = p_alpha-t_alpha + pi/2*j
            tmp_mde[j], _, _, _ = point_wise_mde(predict,target,alpha)

        mde[i] = np.min(tmp_mde)
        alpha = p_alpha-t_alpha + pi/2*np.argmin(tmp_mde)
        _, predict, target, _ = point_wise_mde(predict,target,alpha)

        if plot:
            # plot results
            axes[i].plot(target[:,0],target[:,1],'r.',label='tar.',ms=1)
            axes[i].plot(target[0,0],target[0,1],'r+',label='tar. stp.')
            axes[i].plot(target[10,0],target[10,1],'r*')
            # predict = np.concatenate((predict, predict[0,:].reshape(1,-1)), axis=0)
            axes[i].plot(predict[:,0],predict[:,1],'b.',label='pre.',ms=1)
            axes[i].plot(predict[0,0],predict[0,1],'b+',label='pre. stp.')
            axes[i].plot(predict[10,0],predict[10,1],'b*')
            axes[i].axis('equal')
            if n > 11: axes[i].set_title('{:.6f}'.format(mde[i]), fontsize='medium')
            else: axes[i].set_title('{}_{:.3f}'.format(fns[i],mde[i]), fontsize='medium')


    if plot:
        axes[0].legend(loc='upper left')
        if n > 11: plt.savefig(join(args.save_dir,'{}_{}_testing_results'.format(batch_idx, ep)), bbox_inches='tight')
        else: plt.savefig(join(args.save_dir,'testing_results_{}'.format(ep)), bbox_inches='tight')

        plt.close(fig)


    return np.mean(mde)


def preprocess_testing_data():
    testing_files = sorted(glob.glob(os.path.join(os.getcwd(),
    'testing data/kinematics and dynamics of machinery/xlsx/*.xlsx')))

    fns = []
    p1s = []
    sps = []
    for file in testing_files:
        p1 = pd.read_excel(file).to_numpy()
        fn = file.split('/')[-1].split('.')[0]
        print(fn)
        if fn != 'Pseudo_ellipse':
            p1 = efd_fitting(p1[:,0],p1[:,1],fn,plot=0,N=120)
            p1_normal = curve_normalization_pca2(p1, fn=fn, plot=0, trans=0, rot=0, scal=1)
            sp = plot_fft_plus_power(p1_normal, fn, plot=0)

            fns.append(fn)
            p1s.append(p1)
            sps.append(sp)

    p1s = np.array(p1s)
    sps = np.array(sps)
    print(p1s.shape)
    print(sps.shape)

    dump([fns,p1s,sps], os.path.join(os.getcwd(), 'data', 'testing_data.joblib'))


def preprocess_testing_data_other_papers():
    testing_files = sorted(glob.glob('testing data/other papers/xlsx/examples/*.xlsx'))

    data = []
    idx = 0
    for file in testing_files:
        fn = file.split('/')[-1].split('.')[0]
        cases = pd.read_excel(file).to_numpy()
        if cases.shape[1] == 9:
            for i in range(cases.shape[0]):
                case_n = '{}_{}'.format(fn,i)
                print(idx,case_n)
                p1 = path_gen_open(cases[i,:4], cases[i,4], cases[i,5], cases[i,6], 120, cases[i,7], cases[i,8])

                if is_open_curve(p1):
                    print('open curve')
                    p1_f = bezier_curve(p1[:,0], p1[:,1], case_n, plot=0, nTimes=120)
                else:
                    p1_f = efd_fitting(p1[:,0], p1[:,1], case_n, plot=0, N=120)

                p1_normal = curve_normalization_pca2(p1_f, fn=case_n, plot=0, trans=0, rot=0, scal=1)
                sp = plot_fft_plus_power(p1_normal, case_n, plot=0)

                data.append([case_n,p1[::4],p1_f,sp])
                idx = idx + 1
        else:
            case_n = fn
            print(idx,case_n)
            p1 = cases
            if fn in ['Yu(2007)_0','Ullah(1997)_0']:
                p1_f = efd_fitting(p1[:,0], p1[:,1], case_n, plot=0, N=120)
            elif is_open_curve(p1):
                print('open curve')
                p1_f = bezier_curve(p1[:,0], p1[:,1], case_n, plot=0, nTimes=120)
            else:
                p1_f = efd_fitting(p1[:,0], p1[:,1], case_n, plot=0, N=120)

            p1_normal = curve_normalization_pca2(p1_f, fn=case_n, plot=0, trans=0, rot=0, scal=1)
            sp = plot_fft_plus_power(p1_normal, case_n, plot=0)

            data.append([case_n,p1,p1_f,sp])
            idx = idx + 1

    dump_path = os.path.join(os.getcwd(), 'data', 'testing_data_other_papers.joblib')
    dump(data, dump_path)
    print('dump {}'.format(dump_path))


def preprocess_competitors():
    testing_files = sorted(glob.glob('testing data/other papers/xlsx/competitors/*.xlsx'))

    for file in testing_files:
        fn = file.split('/')[-1].split('.')[0]
        cases = pd.read_excel(file).to_numpy()
        for i in range(cases.shape[0]):
            case_n = '{}_{}'.format(fn,i)
            print(case_n)
            p1 = path_gen_open(cases[i,:4], cases[i,4], cases[i,5], cases[i,6], 120, cases[i,7], cases[i,8])

            if is_open_curve(p1):
                print('open curve')
                p1_f = bezier_curve(p1[:,0], p1[:,1], case_n, plot=0, nTimes=120)
            else:
                p1_f = efd_fitting(p1[:,0], p1[:,1], case_n, plot=1, N=120)

            p1_normal = curve_normalization_pca2(p1_f, fn=case_n, plot=0, trans=0, rot=0, scal=1)
            sp = plot_fft_plus_power(p1_normal, case_n, plot=0)

            data = [case_n,p1,p1_f,sp]
            dump_path = os.path.join(os.getcwd(), 'data', '{}.joblib'.format(case_n))
            dump(data, dump_path)
            print('dump {}'.format(dump_path))


def matching2(nums, links, targets, fn, args, plot=False):

    dirs = {'GCCC':[[0],[1,2,3,5,6]],
            'GCRR':[[1],[0,2,3,5,6]],
            'GRCR':[[2],[0,1,3,5,6]],
            'GRRC':[[3],[0,1,2,5,6]],
            'RRR1':[[1],[0,2,3,5,6]],
            'RRR2':[[0],[1,2,3,5,6]],
            'RRR3':[[0],[2,1,3,5,6]],
            'RRR4':[[0],[3,1,2,5,6]]}
    slice = dirs[args.select_dir]
    n = links.shape[0]
    mde = np.ones(n)
    if plot:
        fig, axes = plt.subplots(1,n,figsize=(4*n,6))

    for i in range(n):
        link = links[i]
        target = targets[i]

        L = np.zeros(9)
        L[slice[0][0]] = 1
        for j in range(len(slice[1])):
            L[slice[1][j]] = link[j]

        predict = path_gen_open(L[:4], L[4], L[5], L[6], target.shape[0], L[7], L[8])
        if predict.shape[0] == 1:
            continue

        # adjust the predicted path to target path
        if args.select_dir in ['GCCC','GCRR']:
            predict = efd_fitting(predict[:,0],predict[:,1],plot=0,N=target.shape[0])
        else:
            predict = bezier_curve(predict[:,0],predict[:,1],plot=0,nTimes=target.shape[0])

        p_alpha, _ = calculate_rotation_angle(predict)
        t_alpha, _ = calculate_rotation_angle(target)

        tmp_mde = np.zeros(4)
        for j in range(4):
            th1 = p_alpha-t_alpha + pi/2*j
            tmp_mde[j], _, _, _ = point_wise_mde(predict,target,th1)

        mde[i] = np.min(tmp_mde)
        th1 = p_alpha-t_alpha + pi/2*np.argmin(tmp_mde)
        _, _, target, ratio = point_wise_mde(predict,target,th1)

        # find the mechanism variables and redraw the curve
        L[:4] *= ratio
        L[4] = -th1
        L[5] *= ratio
        predict = path_gen_open(L[:4], L[4], L[5], L[6], target.shape[0], L[7], L[8])
        if args.select_dir in ['GCCC','GCRR']:
            predict = efd_fitting(predict[:,0],predict[:,1],plot=0,N=target.shape[0])
        else:
            predict = bezier_curve(predict[:,0],predict[:,1],plot=0,nTimes=target.shape[0])
        L[7], L[8] = np.mean(target, axis=0) - np.mean(predict, axis=0)
        p1_mech, p2_mech = path_gen_open_plot_links(L[:4], L[4], L[5], L[6], target.shape[0], L[7], L[8])
        predict_1 = p1_mech[:,2,:]
        predict_2 = p2_mech[:,2,:]

        if plot:
            # plot results
            axes[i].plot(target[:,0],target[:,1],'ro',label='Given',ms=3)
            axes[i].plot(predict_1[:,0],predict_1[:,1],'b',label='Synthesized',ms=3)
            axes[i].plot(predict_2[:,0],predict_2[:,1],'b',ms=3)
            axes[i].axis('equal')
            axes[i].set_title(nums[i], fontsize=12)
            axes[i].set_xlabel('ade = {:.6f}'.format(mde[i]), fontsize=12)

    if plot:
        axes[0].legend(loc='upper left')
        plt.savefig('testing data/{}/{}_testing_results'.format(args.select_dir,fn[1]), bbox_inches='tight')
        plt.close(fig)


    return np.mean(mde), L


def matching3(links, targets, gt, fn, args, plot=False, compete=True):

    dirs = {'GCCC':[[0],[1,2,3,5,6]],
            'GCRR':[[1],[0,2,3,5,6]],
            'GRCR':[[2],[0,1,3,5,6]],
            'GRRC':[[3],[0,1,2,5,6]],
            'RRR1':[[1],[0,2,3,5,6]],
            'RRR2':[[0],[1,2,3,5,6]],
            'RRR3':[[0],[2,1,3,5,6]],
            'RRR4':[[0],[3,1,2,5,6]]}
    slice = dirs[args.select_dir]
    n = links.shape[0]
    mde = np.ones(n)
    if plot:
        fig, axes = plt.subplots(1,n*2,figsize=(8*n,8))
        plt.rcParams.update({'font.size': 12})

    for i in range(n):
        link = links[i]
        target = targets[i]

        L = np.zeros(9)
        L[slice[0][0]] = 1
        for j in range(len(slice[1])):
            L[slice[1][j]] = link[j]

        predict = path_gen_open(L[:4], L[4], L[5], L[6], target.shape[0], L[7], L[8])
        if predict.shape[0] == 1:
            continue

        # adjust the predicted path to target path
        if args.select_dir in ['GCCC','GCRR']:
            predict = efd_fitting(predict[:,0],predict[:,1],plot=0,N=target.shape[0])
        else:
            predict = bezier_curve(predict[:,0],predict[:,1],plot=0,nTimes=target.shape[0])

        p_alpha, _ = calculate_rotation_angle(predict)
        t_alpha, _ = calculate_rotation_angle(target)

        tmp_mde = np.zeros(4)
        for j in range(4):
            th1 = p_alpha-t_alpha + pi/2*j
            tmp_mde[j], _, _, _ = point_wise_mde(predict,target,th1)

        mde[i] = np.min(tmp_mde)
        th1 = p_alpha-t_alpha + pi/2*np.argmin(tmp_mde)
        _, _, target, ratio = point_wise_mde(predict,target,th1)

        # find the mechanism variables and redraw the curve
        L[:4] *= ratio
        L[4] = -th1
        L[5] *= ratio
        predict = path_gen_open(L[:4], L[4], L[5], L[6], target.shape[0], L[7], L[8])
        if args.select_dir in ['GCCC','GCRR']:
            predict = efd_fitting(predict[:,0],predict[:,1],plot=0,N=target.shape[0])
        else:
            predict = bezier_curve(predict[:,0],predict[:,1],plot=0,nTimes=target.shape[0])
        L[7], L[8] = np.mean(target, axis=0) - np.mean(predict, axis=0)
        p1_mech, p2_mech = path_gen_open_plot_links(L[:4], L[4], L[5], L[6], target.shape[0], L[7], L[8])
        if args.select_dir == 'GRRC':
            p2_mech = np.flip(p2_mech,axis=0)
            p_mech = np.concatenate((p1_mech, p2_mech), axis=0)
        else:
            p_mech = p1_mech
        predict = p_mech[:,2,:]


        if plot:
            if compete:
                cited_num = '[35]'
                competitor_root = os.path.join('data', '{}.joblib'.format(fn))
                _,comp,_,_ = load(competitor_root)
                # _, _, comp, _ = point_wise_mde(comp,target,th1) # apply when test Khan's examples

            axes[0].plot(gt[:,0],gt[:,1],'ro',mfc='none',label='Desired',ms=5)
            axes[0].plot(predict[:,0],predict[:,1],'b',label='Proposed method',ms=3)
            if compete:
                axes[0].plot(comp[:,0],comp[:,1],'k--',label=fn[:-16]+cited_num,ms=1)
            axes[0].axis('equal')
            axes[0].legend(loc='upper left')

            # fix the x, y limits
            xmin, ymin = np.min(p_mech.reshape(-1,2),axis=0)
            xmax, ymax = np.max(p_mech.reshape(-1,2),axis=0)
            xmin = xmin-abs(xmax-xmin)*0.1
            ymin = ymin-abs(ymax-ymin)*0.1
            xmax = xmax+abs(xmax-xmin)*0.1
            ymax = ymax+abs(ymax-ymin)*0.1

            fps = 2
            step = int(p_mech.shape[0]/fps)
            for j in range(0,p_mech.shape[0],step):
                axes[1].plot(gt[:,0],gt[:,1],'ro',mfc='none',label='Desired',ms=5)
                axes[1].plot(predict[:,0],predict[:,1],'b',label='Proposed method',ms=3)
                if compete:
                    axes[1].plot(comp[:,0],comp[:,1],'k--',label=fn[:-16]+cited_num,ms=1)
                axes[1].axis('equal')

                # plot mechanism
                dots = p_mech[j]
                axes[1].plot(dots[:,0],dots[:,1],'k',ms=1)
                axes[1].plot(dots[:,0],dots[:,1],'ko',mfc='none',ms=6)
                axes[1].plot(dots[0,0],dots[0,1],'ko',ms=6)
                axes[1].plot(dots[4,0],dots[4,1],'ko',ms=6)

                # axes[1].legend(loc='upper left')
                axes[1].set_xlim(xmin, xmax)
                axes[1].set_ylim(ymin, ymax)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                interval = 360/p_mech.shape[0]
                save_path = 'testing data/other papers/{}_testing_results_{}'.format(fn,int(j*interval))
                plt.savefig(save_path, bbox_inches='tight')
                axes[1].clear()


    if plot:
        plt.close(fig)


    return np.mean(mde), L


def QJGe_example_ad(case='a'):

    example = []
    if case == 'a':
        range = np.linspace(-0.5*pi, 0.5*pi, 120, dtype=np.complex128)
        for wt in range:
            T = (100+20j)+(20+14j)*exp(-1j*wt)+(60-30j)*exp(1j*wt)+(2+4j)*exp(-2j*wt)+(9-5j)*exp(2j*wt)
            print(T.real, T.imag)
            example.append([T.real, T.imag])

    elif case == 'd':
        range = np.linspace(0.5, 0.93, 120)
        for x in range:
            y = 0.25/0.43*(x-0.5)
            print(x, y)
            example.append([x, y])
    example = np.array(example)

    fig, axes = plt.subplots(1,1,figsize=(4,4))
    axes.plot(example[:,0],example[:,1],'ko')
    axes.axis('equal')
    plt.show()


def main():
    #----------------------- CREATE A RANDOM PATH -------------------------------
    num = [1215,514,1225,120]
    seed = num[3]
    np.random.seed(seed)
    N = 60
    example_num = 1
    Ls, _, _ = fourbar_mechanism(inversion='GCRR', set=1, N=N)
    L = Ls[example_num]
    # L[0] = 76.77
    # L[1] = 27.31
    # L[2] = 46.44
    # L[3] = 72.55
    # L[5] = 60.42
    # L[6] = 314.5/180*3.1415926
    print(L[:4])

    p1 = path_gen_open(L[:4], L[4], L[5], L[6], N, L[7], L[8])
    p1 = efd_fitting(p1[:,0],p1[:,1],fn='',plot=1, N=120)
    # p1 = bezier_curve(p1[:,0], p1[:,1], fn='', plot=0, nTimes=120)

    fn = "initial_{}".format(seed)
    print("-----Initial path-----")
    # p1_init_normal = curve_normalization_pca2(p1, fn, plot=0)

    #------------------------ NORMALIZATION METHODS ---------------------------
    # print("-----Testing path-----")
    ## curve_normalization_pca2
    p1_normal = curve_normalization_pca2(p1, fn=fn, plot=0, trans=0, rot=0, scal=1)
    # p1_normal = curve_normalization_pca2(p1_normal, fn="2nd normalization", plot=0)

    # print("Normalized error: {:.3f}".format(LA.norm(p1_init_normal-p1_normal)))
    # fig, ax = plt.subplots(figsize=(8,4))
    # ax.plot(p1_init_normal[:,0],p1_init_normal[:,1],label='init_normalized')
    # ax.plot(p1_normal[:,0],p1_normal[:,1],label='various_normalized')
    # ax.legend(loc='upper left')
    # plt.axis('equal')
    # plt.savefig('./test_figure/curve_normalization_pca/init_var_normal.png')

    #------------------------ PATH TRANSFORMATION ---------------------------
    sp = plot_fft_plus_power(p1_normal, fn, plot=0)

    #----------------------- Intersected and United --------------------------
    # p2 = grashof(Inversion='GCRR', set=2, N=120)
    # p2 = efd_fitting(p2[:,0],p2[:,1],plot=0)
    # p2_normal = curve_normalization_pca2(p2, fn=fn, plot=0, trans=0, rot=0, scal=1)
    # IOU(p1_normal, p2_normal)


if __name__ == '__main__':
    # main()
    # preprocess_testing_data()
    preprocess_testing_data_other_papers()
    # preprocess_competitors()
    # QJGe_example_ad(case='a')
