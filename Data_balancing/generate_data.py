import numpy as np
import cv2
import os
import sys

from matplotlib import pyplot as plt
from matplotlib import _color_data as mcd
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from joblib import dump, load
from scipy import ndimage
from itertools import permutations
from tqdm import tqdm

from clustering import load_data
from preprocessing import fourbar_mechanism, efd_fitting, curve_normalization_pca2, plot_fft_plus_power

colormap = [name for name in mcd.CSS4_COLORS
           if "xkcd:" + name in mcd.XKCD_COLORS]
colormap.remove('white')
colormap = sorted(colormap)

testing_data_name = [
'Banana','Crescent','Double_straight',
'Figure_eight','Kidney_bean',
'Scimitar','Single_straight','Teardrop',
'Triple_cusps','Triple_loops','Umbrella']


def load_pca_kmeans(inversion, pca_n_dim, n_clusters):
    predict_model = 108576
    load_path = os.path.join('clusters/pca', inversion, 'pca_{}_{}.joblib'.format(predict_model,pca_n_dim))
    pca = load(os.path.join(os.getcwd(), load_path))
    print('Load '+ load_path)

    load_path = os.path.join('clusters/kmeans', inversion, 'kmeans_{}_{}_{}.joblib'.format(predict_model,pca_n_dim,n_clusters))
    kmeans = load(os.path.join(os.getcwd(), load_path))
    print('Load '+ load_path)

    return pca, kmeans


def balanced_data_generator(seed, inversion, data_size, pca_n_dim, set, train_sps, n_clusters):
    # Load models
    pca, kmeans = load_pca_kmeans(inversion, pca_n_dim, n_clusters)
    train_sps = pca.transform(train_sps)
    scaler = StandardScaler()
    scaler.fit(train_sps)

    if data_size == 12896: total_set = 4
    elif data_size == 2496: total_set = 1

    n_data_cluster = int(data_size / n_clusters / total_set)

    data_L = [[] for i in range(n_clusters)]
    data_p1s = [[] for i in range(n_clusters)]
    data_sps = [[] for i in range(n_clusters)]
    data_num = [0 for i in range(n_clusters)]
    while(min(data_num)<n_data_cluster):
        b_min_data_num = min(data_num)
        # Transform and predict labels of new data
        L, p1s, sps = fourbar_mechanism(inversion=inversion, set=1, n=data_size, N=120)

        pca_sps = pca.transform(sps)
        pca_sps = scaler.transform(pca_sps)
        predict_label = kmeans.predict(pca_sps)

        for i in range(n_clusters):
            tmp_L = L[i==predict_label].tolist()
            if len(data_L[i])+len(tmp_L) > n_data_cluster:
                req_num = n_data_cluster - len(data_L[i])
                for j in range(req_num):
                    data_L[i].append(tmp_L[j])
            else:
                for j in tmp_L:
                    data_L[i].append(j)

            tmp_p1s = p1s[i==predict_label].tolist()
            if len(data_p1s[i])+len(tmp_p1s) > n_data_cluster:
                req_num = n_data_cluster - len(data_p1s[i])
                for j in range(req_num):
                    data_p1s[i].append(tmp_p1s[j])
            else:
                for j in tmp_p1s:
                    data_p1s[i].append(j)

            tmp_sps = sps[i==predict_label].tolist()
            if len(data_sps[i])+len(tmp_sps) > n_data_cluster:
                req_num = n_data_cluster - len(data_sps[i])
                for j in range(req_num):
                    data_sps[i].append(tmp_sps[j])
            else:
                for j in tmp_sps:
                    data_sps[i].append(j)

            data_num[i]=len(data_L[i])
        print("{} / {}".format(min(data_num), n_data_cluster))

    data_L = np.array(data_L)
    data_p1s = np.array(data_p1s)
    data_sps = np.array(data_sps)

    data_L = data_L.reshape(-1,data_L.shape[-1])
    data_p1s = data_p1s.reshape(-1,data_p1s.shape[-2],data_p1s.shape[-1])
    data_sps = data_sps.reshape(-1,data_sps.shape[-1])

    save_name = 'data_b_{}_{}_{}_{}.joblib'.format(seed,data_size,n_clusters,set)
    dump_path = os.path.join('data', inversion, 'data_b_1215_{}_{}'.format(pca_n_dim,data_size),save_name)
    dump([data_L,data_p1s,data_sps], os.path.join(os.getcwd(), dump_path))
    print('output ' + save_name)


def merge_data(inversion, seed, data_size, n_clusters):

    data_name = 'data_b_{}_{}_{}_{}.joblib'.format(seed,data_size,n_clusters,0)
    load_path = os.path.join('data', inversion, 'data_b_1215_2_{}'.format(data_size), data_name)
    data = load(os.path.join(os.getcwd(), load_path))
    L, p1s, sps = data[0], data[1], data[2]
    # print(L[0,:4])
    max = 4 if data_size==14560 or data_size==12896 else 1

    for i in range(1,max):
        data_name = 'data_b_{}_{}_{}_{}.joblib'.format(seed,data_size,n_clusters,i)
        load_path = os.path.join('data', inversion, 'data_b_1215_2_{}'.format(data_size), data_name)
        data = load(os.path.join(os.getcwd(), load_path))
        tmp_L, tmp_p1s, tmp_sps = data[0], data[1], data[2]
        # print(tmp_L[0,:4])

        L = np.concatenate((L, tmp_L), axis=0)
        p1s = np.concatenate((p1s, tmp_p1s), axis=0)
        sps = np.concatenate((sps, tmp_sps), axis=0)

        print(L.shape,p1s.shape,sps.shape,"\n")

    # sys.exit()
    save_name = 'data_b_{}_{}_{}.joblib'.format(seed,data_size,n_clusters)
    dump([L,p1s,sps], os.path.join(os.getcwd(),'data', inversion, save_name))


def inverse_scaling(seed, data_size, n_clusters, scale):
    data_name = 'data_b_{}_{}_{}.joblib'.format(seed,data_size,n_clusters)
    data = load(os.path.join(os.getcwd(), 'data', 'scaled', data_name))
    L, p1s, sps = data[0], data[1], data[2]
    sps = sps*scale[1] + scale[0]

    save_name = 'data_b_{}_{}_{}.joblib'.format(seed,data_size,n_clusters)
    dump([L,p1s,sps], os.path.join(os.getcwd(),'data', save_name))


def plot_distribution(seed, inversion, data_size, n_clusters, scale, type='random'):
    # load pca and kmeans models
    pca, kmeans = load_pca_kmeans(inversion, pca_n_dim, n_clusters)

    # load data to see distribution
    if type == 'balanced':
        data_name = 'data_b_{}_{}_{}.joblib'.format(seed,data_size,n_clusters)
        t_data_name = 'testing_data.joblib'
        title = 'kmeans_b_{}_{}_wt'.format(data_size,n_clusters)
    elif type == 'random':
        data_name = 'data_{}_{}.joblib'.format(seed,data_size)
        t_data_name = 'testing_data.joblib'
        title = 'kmeans_{}_{}_wt'.format(data_size,n_clusters)
    elif type == 'exhausted':
        data_name = 'data_e_{}.joblib'.format(data_size)
        t_data_name = 'testing_data.joblib'
        title = 'kmeans_e_{}_{}_wt'.format(data_size,n_clusters)

    data = load(os.path.join(os.getcwd(), 'data', inversion, data_name))
    _, _, sps = data[0], data[1], data[2]
    print("Load ", data_name)
    sps = (sps - scale[0]) / scale[1]

    # t_data = load(os.path.join(os.getcwd(), 'data', t_data_name))
    # _, t_sps = t_data[0], t_data[1]
    # print("Load ", t_data_name)

    pca_sps = pca.transform(sps)
    predict_label = kmeans.predict(pca_sps)

    # t_pca_sps = pca.transform(t_sps)
    # t_predict_label = kmeans.predict(t_pca_sps)

    fig, ax = plt.subplots(figsize=(8,8))
    for i in range(n_clusters):
        points = pca_sps[predict_label == i]
        label = str(i)+'_{}'.format(points.shape[0])
        color = mcd.CSS4_COLORS[colormap[i]]
        ax.scatter(points[:,0], points[:,1], c=color, label=label)

    # for i in range(len(testing_data_name)):
    #     t_points = t_pca_sps[i]
    #     t_label = str(t_predict_label[i])+'_{}'.format(testing_data_name[i])
    #     ax.plot(t_points[0], t_points[1],'r+' , label=t_label)

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend(loc='upper left')
    save_path = os.path.join(os.getcwd(),'clusters', 'kmeans', title)
    plt.savefig(save_path)
    print('output path:{}'.format(save_path))


def generate_all_balanced_data():
    inversions = [
                  # ['GCCC',16],
                  # ['GCRR',16],
                  # ['GRCR',16],
                  # ['GRRC',16],
                  # ['RRR1',[16,18]],
                  # ['RRR2',[18,14,17,16]],
                  # ['RRR3',[9]],
                  ['RRR4',[7]],
                 ]
    num = [1215,514,1225,120]
    seed = num[0]
    data_set = 'random'
    pca_n_dim = 2

    balance_data_size = 2496    # 0.25 -> 189280,108576 0.5 -> 14560,12896, 0.9 -> 2496
    set = 1

    for (inversion,cluster_numbers) in inversions:
        # Load feature scaling data
        # _, _, train_sps = load_data(inversion, data_set, seed=seed, data_size=108576)

        for n_clusters in cluster_numbers:
            # balanced_data_generator(seed, inversion, balance_data_size, pca_n_dim, set, train_sps, n_clusters)
            merge_data(inversion, seed, balance_data_size, n_clusters)


def main():
    # ========================================================================
    # Set up initial values
    num = [1215,514,1225,120]
    seed = num[0]

    inversion = 'RRR1'
    data_set = 'random'

    data_size = 2496    # 0.25 -> 189280, 0.5 -> 14560,12896 0.9 -> 2496
    pca_n_dim = 2
    set = 0

    # Load feature scaling data
    # _, _, train_sps = load_data(inversion, data_set, seed=1215, data_size=108576)

    for n_clusters in range(2,16):
        # balanced_data_generator(seed, inversion, data_size, pca_n_dim, set, train_sps, n_clusters)
        merge_data(inversion, seed, data_size, n_clusters)
        # plot_distribution(seed, inversion, data_size, n_clusters, scale, type='random')



if __name__ == '__main__':
    # main()
    generate_all_balanced_data()
