import numpy as np
import cv2
import os
import sys

from matplotlib import pyplot as plt
from matplotlib import _color_data as mcd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.manifold import TSNE
from sklearn import metrics
from math import cos
from joblib import dump, load
from scipy import ndimage
from itertools import permutations
from tqdm import tqdm

from preprocessing import fourbar_mechanism, efd_fitting, bezier_curve, curve_normalization_pca2, plot_fft_plus_power
from utils import cm_sets


def load_data(inversion, dataset='random', seed=1215, data_size=189280, n_clusters=3):
    if dataset == 'random':
        data_name = 'data_{}_{}.joblib'.format(seed,data_size)
    elif dataset == 'exhaustive':
        data_name = 'data_e_{}.joblib'.format(data_size)
    elif dataset == 'balanced':
        data_name = 'data_b_{}_{}_{}.joblib'.format(seed,data_size,n_clusters)

    try:
        load_path = os.path.join('data', inversion, data_name)
        data = load(os.path.join(os.getcwd(), load_path))
        L, p1s, sps = data[0], data[1], data[2]
        print("Load ", load_path)

    except FileNotFoundError:
        print(data_name, " does not exist.")
        if dataset != 'balanced':
            L, p1s, sps = fourbar_mechanism(inversion=inversion, set=1, n=data_size, N=120)

        if dataset == 'random':
            dump_path = os.path.join('data', inversion, data_name)
            dump([L,p1s,sps], os.path.join(os.getcwd(), dump_path))
            print("Dump ", dump_path)

        elif dataset == 'exhaustive':
            dump_path = os.path.join('data', inversion, 'data_e_{}.joblib'.format(L.shape[0]))
            dump([L,p1s,sps], os.path.join(os.getcwd(), dump_path))
            print("Dump ", dump_path)

    print(L.shape, p1s.shape, sps.shape)
    return L, p1s, sps


def load_pca(dataset, inversion, sps, pca_n_dim, n_clusters):
    if dataset == "random":
        model_name = 'pca_{}_{}.joblib'.format(sps.shape[0],pca_n_dim)
    elif dataset == "balanced":
        if 'G' in inversion:
            data_size = 189280
        else:
            data_size = 108576
        model_name = 'pca_{}_{}.joblib'.format(data_size,pca_n_dim)
        # model_name = 'pca_{}_{}_{}.joblib'.format(sps.shape[0],pca_n_dim,n_clusters)

    try:
        pca = load(os.path.join(os.getcwd(), 'clusters', 'pca', inversion, model_name))
        print('Load {}...'.format(model_name))
        print(np.sum(pca.explained_variance_ratio_))
        pca_sps = pca.transform(sps)

    except FileNotFoundError:
        print('pca dimensionality reduction...')
        pca = PCA(n_components=pca_n_dim,
                  svd_solver='auto',
                  whiten=False).fit(sps)
        dump(pca, os.path.join(os.getcwd(), 'clusters', 'pca', inversion, model_name))
        pca_sps = pca.transform(sps)
        print(np.sum(pca.explained_variance_ratio_))
        print('Dump {}...'.format(model_name))

    return pca_sps


def load_kmeans(dataset, inversion, pca_sps, pca_n_dim, n_clusters):

    model_name = 'kmeans_{}_{}_{}.joblib'.format(pca_sps.shape[0],pca_n_dim,n_clusters)

    try:
        kmeans = load(os.path.join(os.getcwd(), 'clusters', 'kmeans', inversion, model_name))
        print('Load {}...'.format(model_name))

    except FileNotFoundError:
        print('kmeans clustering...')
        kmeans = KMeans(n_clusters=n_clusters,
                        init='k-means++',
                        n_init=100,
                        max_iter=3000,
                        tol=5e-5,
                        precompute_distances='auto',
                        verbose=0,
                        random_state=514,
                        copy_x=True,
                        n_jobs=5,
                        algorithm='auto').fit(pca_sps)
        dump(kmeans, os.path.join(os.getcwd(), 'clusters', 'kmeans', inversion, model_name))
        print('Dump {}...'.format(model_name))

    return kmeans


def load_tsne(dataset, inversion, sps, tsne_seed, tsne_size, n_clusters):
    if dataset == "random":
        model_name = 'tsne_{}_{}_{}_{}.joblib'.format(sps.shape[0],sps.shape[1],tsne_seed,tsne_size)
    elif dataset == "balanced":
        model_name = 'tsne_{}_{}_{}_{}_{}.joblib'.format(sps.shape[0],sps.shape[1],tsne_seed,tsne_size,n_clusters)

    try:
        np.random.seed(tsne_seed)
        tsne_sps, rand_ind = load(os.path.join(os.getcwd(), 'clusters', 'tsne', inversion, model_name))
        print('Load {}...'.format(model_name))

    except FileNotFoundError:
        print('tsne dimensionality reduction...')
        rand_ind = np.random.randint(len(sps), size=tsne_size)
        tsne_sps = sps[rand_ind]
        tsne_sps = TSNE(n_components=2, random_state=0).fit_transform(tsne_sps)
        dump([tsne_sps,rand_ind], os.path.join(os.getcwd(), 'clusters', 'tsne', inversion, model_name))
        print('Dump {}...'.format(model_name))

    return tsne_sps, rand_ind


def chi_square_uniformity(sps, dataset="random", inversion='GCRR', n_clusters=12, pca_n_dim=2):
    # load pca models
    pca_sps = load_pca(dataset, inversion, sps, pca_n_dim, n_clusters)

    # feature scaling
    scaler = StandardScaler()
    if dataset == "balanced":
        if 'G' in inversion:
            data_size = 189280
        else:
            data_size = 108576
        _, _, train_sps = load_data(inversion, "random", 1215, data_size)
        train_sps = load_pca("random", inversion, train_sps, pca_n_dim, n_clusters)
        scaler.fit(train_sps)
    else:
        scaler.fit(pca_sps)

    pca_sps = scaler.transform(pca_sps)

    # find the boundary
    n_std = 3
    xmin = -n_std
    xmax = n_std
    ymin = -n_std
    ymax = n_std

    # set number of patches
    div = 10
    x_tick = np.linspace(xmin,xmax,div+1)
    y_tick = np.linspace(ymax,ymin,div+1)

    # find dots per patch
    dpp = []
    for i in range(1,div+1):
        patch_y_min = y_tick[i]
        patch_y_max = y_tick[i-1]
        for j in range(1,div+1):
            patch_x_min = x_tick[j-1]
            patch_x_max = x_tick[j]
            dots = pca_sps[pca_sps[:,1]>patch_y_min]
            dots = dots[dots[:,1]<patch_y_max]
            dots = dots[dots[:,0]>patch_x_min]
            dots = dots[dots[:,0]<patch_x_max]

            # color = colors[0]
            # ax.scatter(dots[:,0], dots[:,1], c=color, s=5)
            dpp.append(dots.shape[0])

    # calculate chi square
    dpp = np.array(dpp)
    diff = (dpp - dpp.mean())**2
    chi_square = diff.sum() / dpp.mean() / pca_sps.shape[0]

    # plot the results
    fig, ax = plt.subplots(1,figsize=(8,8))
    plt.rcParams.update({'font.size': 16})
    if dataset == "balanced":
        ax.set_title('{}, k = {}, Chi-Square = {:.4f}'.format(inversion, n_clusters, chi_square))
    else:
        ax.set_title('{}, Chi-Square = {:.4f}'.format(inversion, chi_square))

    ax.set_xlabel('X',fontsize=16)
    ax.set_ylabel('Y',fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # ax.imshow(dpp.reshape(div,div), cmap='Reds')
    y, x = np.meshgrid(np.linspace(xmin, xmax, div), np.linspace(ymin, ymax, div))
    dpp = dpp.reshape(div,div) / pca_sps.shape[0]
    c = ax.pcolormesh(x, y, dpp, cmap='Reds', vmin=dpp.min(), vmax=dpp.max())
    ax.axis([xmin, xmax, ymin, ymax])
    fig.colorbar(c, ax=ax)

    if dataset == "balanced":
        save_name = 'chi_square_{}_{}_2_1566_{}'.format(inversion,pca_sps.shape[0],n_clusters)
    else:
        save_name = 'chi_square_{}_{}_2_1566'.format(inversion,pca_sps.shape[0])

    save_path = os.path.join(os.getcwd(),'clusters/chi_square',inversion,save_name)
    plt.savefig(save_path, bbox_inches='tight')
    print('output path:{}'.format(save_path))
    plt.close()

    return chi_square


def kmeans_clustering(sps, dataset="random", inversion='GCRR', n_clusters=12, pca_n_dim=4, tsne_seed=1215, tsne_size=10000, plot=0):

    # load pca models
    pca_sps = load_pca(dataset, inversion, sps, pca_n_dim, n_clusters)

    # feature scaling
    scaler = StandardScaler()
    pca_sps = scaler.fit_transform(pca_sps)

    # load kmeans models
    kmeans = load_kmeans(dataset, inversion, pca_sps, pca_n_dim, n_clusters)
    # print(kmeans.cluster_centers_)

    # load tsne sps
    tsne_sps, rand_ind = load_tsne(dataset, inversion, sps, tsne_seed, tsne_size, n_clusters)

    labels = kmeans.labels_
    tsne_labels = labels[rand_ind]

    # using pca_sps to plot
    tsne_sps = pca_sps[rand_ind]
    n_std = 3

    for i in range(2):
        tsne_labels = tsne_labels[tsne_sps[:,i]<n_std]
        tsne_sps = tsne_sps[tsne_sps[:,i]<n_std]
        tsne_labels = tsne_labels[tsne_sps[:,i]>-n_std]
        tsne_sps = tsne_sps[tsne_sps[:,i]>-n_std]

    tsne_sps = scaler.inverse_transform(tsne_sps)


    # evaluation of clustering results
    # s_score = metrics.silhouette_score(pca_sps, labels, metric='euclidean')
    # ch_score = metrics.calinski_harabaz_score(pca_sps, labels)
    db_score = metrics.davies_bouldin_score(pca_sps, labels)
    scores = [db_score]
    print(scores)


    if plot:
        # sort the labels depending on quantity in each clusters
        cluster_num = len(np.unique(labels))
        labels_num = []
        for i in range(cluster_num):
            path_ind = np.where(labels == i)[0]
            labels_num.append([i,len(path_ind)/sps.shape[0]])
        labels_num = np.array(labels_num)
        labels_num = labels_num[labels_num[:,1].argsort()]
        labels_num = labels_num[::-1]

        fig, axes = plt.subplots(1,figsize=(8,8))
        colors = cm_sets()
        plt.rcParams.update({'font.size': 12})
        for i in range(cluster_num):
            points = tsne_sps[tsne_labels == labels_num[i,0]]
            label = str(int(labels_num[i,0]))+'_{:.4f}'.format(labels_num[i,1])
            color = colors[i]
            axes.scatter(points[:,0], points[:,1], c=color, label=label, s=5)

        axes.set_title('{}, k = {}'.format(inversion, n_clusters))
        axes.set_xlabel('X',fontsize=12)
        axes.set_ylabel('Y',fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        axes.axis('equal')
        axes.legend(loc='upper left')
        save_path = os.path.join(os.getcwd(),'clusters/kmeans',inversion,'kmeans_results', 'kmeans_{}_{}_{}_{}_{}'
                                 .format(inversion,sps.shape[0],pca_n_dim,tsne_seed,n_clusters))
        plt.savefig(save_path, bbox_inches='tight')
        print('output path:{}'.format(save_path))

        plt.close(fig)


    return labels, scores


def mean_shift_clustering(sps, bandwidth=0.1, plot=0):
    # The following bandwidth can be automatically detected using
    if bandwidth == 0:
        bandwidth = estimate_bandwidth(sps, quantile=0.2)

    try:
        pca = load(os.path.join(os.getcwd(), 'clusters', 'mean_shift', 'pca_{}.joblib'.format(sps.shape[0])))
        print('Load pca_{}.joblib...'.format(sps.shape[0]))
        reduced_sps = pca.transform(sps)

    except FileNotFoundError:
        print('PCA dimensionality reduction...')
        pca = PCA(n_components=2,svd_solver='auto',whiten=False).fit(sps)
        dump(pca, os.path.join(os.getcwd(), 'clusters', 'mean_shift', 'pca_{}.joblib'. format(sps.shape[0])))
        reduced_sps = pca.transform(sps)

    try:
        ms = load(os.path.join(os.getcwd(), 'clusters', 'mean_shift', 'ms_{}_{}.joblib'.
             format(sps.shape[0],bandwidth)))
        print('Load ms_{}_{}.joblib...'.format(sps.shape[0],bandwidth))

    except FileNotFoundError:
        print('Mean shift clustering...')
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=5).fit(sps)
        dump(ms, os.path.join(os.getcwd(), 'clusters', 'mean_shift', 'ms_{}_{}.joblib'.
             format(sps.shape[0],bandwidth)))

    labels = ms.labels_
    centers = ms.cluster_centers_

    n_cluster = len(np.unique(labels))
    labels_num = []
    for i in range(n_cluster):
        path_ind = np.where(labels == i)[0]
        labels_num.append([i,len(path_ind)])
    labels_num = np.array(labels_num)
    labels_num = labels_num[labels_num[:,1].argsort()]
    print("number of estimated clusters : {}".format(n_cluster))

    if plot:
        _, axes = plt.subplots(1,2,figsize=(8,8))
        for i in range(n_cluster):
            points = reduced_sps[labels == labels_num[i,0]]
            label = str(i)+'_{}'.format(points.shape[0])
            center = centers[labels_num[i,0]]
            if i >= 7:
                axes[0].scatter(points[:,0], points[:,1], c=[plt.cm.Set1(i-7)])
                axes[1].scatter(center[0], center[1], c=[plt.cm.Set1(i-7)], label=label)
            else:
                axes[0].scatter(points[:,0], points[:,1], c=[plt.cm.Set2(i)])
                axes[1].scatter(center[0], center[1], c=[plt.cm.Set2(i)], label=label)

        axes[0].set_title('Samples')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        axes[1].set_title('Centers')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Y')
        axes[1].legend(loc='upper left')
        save_path = os.path.join(os.getcwd(),'clusters', 'mean_shift', 'pca_{}_{}'.format(sps.shape[0],n_cluster))
        plt.savefig(save_path)
        print('output path:{}'.format(save_path))

    return labels, centers


def plot_cluster_result(inversion, p1s, labels, pca_n_dim, tsne_seed, L, rows=10):
    n_cluster = len(np.unique(labels))

    sorted_labels = []
    for i in range(n_cluster):
        path_ind = np.where(labels == i)[0]
        sorted_labels.append([i,len(path_ind)])
    sorted_labels = np.array(sorted_labels)
    sorted_labels = sorted_labels[sorted_labels[:,1].argsort()]
    sorted_labels = sorted_labels[::-1]

    cluster_path = []
    cluster_L = []
    for i in range(n_cluster):
        path_ind = np.where(labels == sorted_labels[i,0])[0]
        rand_path_ind = np.random.randint(len(path_ind), size=rows)
        cluster_path.append(p1s[path_ind[rand_path_ind]])
        cluster_L.append(L[path_ind[rand_path_ind]])
    # col = 0
    # row = 4
    # print(cluster_L[col][row])
    # plt.scatter(cluster_path[col][row][:,0],cluster_path[col][row][:,1])
    # plt.axis('equal')
    # plt.show()
    # sys.exit()
    # plot the first ten input images and then reconstructed images
    _, axes = plt.subplots(nrows=rows, ncols=n_cluster, figsize=(40,35))
    colors = cm_sets()
    plt.rcParams.update({'font.size': 18})
    for col in range(n_cluster):
        axes[0][col].set_title('{}_{}'.format(sorted_labels[col,0], sorted_labels[col,1]))
        color = colors[col]
        for row in range(rows):
            p1 = cluster_path[col][row]
            axes[row][col].scatter(p1[:,0],p1[:,1],c=color)
            axes[row][col].axis('equal')

    save_path = os.path.join(os.getcwd(), 'clusters/kmeans', inversion, 'cluster_results',
                            'cluster_result_{}_{}_{}_{}_{}'.format(inversion,p1s.shape[0], pca_n_dim, tsne_seed, n_cluster))
    plt.savefig(save_path, bbox_inches='tight')
    print('output path:{}'.format(save_path))


def find_kmeans_elbow_point(sps, pca_n_dim, tsne_seed):
    inertias = []
    for i in range(1,30):
        n_clusters = i

        labels, _, inertia = kmeans_clustering(sps,
        n_clusters=n_clusters,
        pca_n_dim=pca_n_dim,
        tsne_seed=tsne_seed,
        tsne_size=10000,
        plot=0)
        inertias.append(inertia/labels.shape[0])

    inertias = np.array(inertias)
    inertias = np.abs(inertias-inertias[0])

    fig, axes = plt.subplots(1,figsize=(8,8))
    axes.plot(inertias)
    axes.set_title('kmeans variance difference, k=2->30')
    axes.set_xlabel('n_clusters')
    axes.set_ylabel('variance difference')
    save_path = os.path.join(os.getcwd(), 'clusters/kmeans', 'kmeans_{}_{}_{}_{}to{}'
                             .format(sps.shape[0],pca_n_dim,tsne_seed,2,30))
    plt.savefig(save_path)
    print('output path:{}'.format(save_path))

    plt.close(fig)


def generate_all_fourbar_mechanism_data():
    # Set up initial values
    num = [1215,514,1225,120]
    seed = num[0]
    np.random.seed(seed)

    inversions = [
                  # 'GCCC',
                  # 'GCRR',
                  # 'GRCR',
                  # 'GRRC'
                  # 'RRR1',
                  # 'RRR2',
                  'RRR3',
                  'RRR4',
                 ]
    # Grashof [0.9, 2496, 0.5, 14560, 0.25, 189280]
    # data_sizes = [0.9, 2496, 0.5, 14560, 0.25, 189280]

    # non-Grashof  [0.8, 2496, 0.5, 12896, 0.3, 108576]
    data_sizes = [108576]

    data_sets = ['exhaustive', 'random']

    # ========================================================================
    # Load data
    for inversion in inversions:
        for size in data_sizes:
            print('\n'+inversion, size)
            if size < 3:
                _, _, _ = load_data(inversion, data_sets[0], seed, size)
            else:
                _, _, _ = load_data(inversion, data_sets[1], seed, size)


def main():
    # ========================================================================
    # Set up initial values
    num = [1215,514,1225,120]
    seed = num[0]
    np.random.seed(seed)

    inversion = 'RRR4'
    data_size = 108576    # 0.25 -> 189280,108576 0.5 -> 14560,12896 0.9 -> 2496
    best_clusters = [
    ('GCCC',14560,11),
    ('GCRR',14560,7),
    ('GRCR',14560,3),
    ('GRRC',14560,4),
    ('RRR1',12896,4),
    ('RRR2',12896,6),
    ('RRR3',12896,9),
    ('RRR4',12896,4),]
    data_set = 'balanced'

    n_clusters = 2      # which includes data, model name and the number we want to cluster
    pca_n_dim = 2
    tsne_seed = 1566
    tsne_size = 10000

    plot = True

    print('random seed number {}, data size {}, dataset {}'.format(seed, data_size, data_set))

    # ========================================================================
    # Load data
    # L, p1s, sps = load_data(inversion, data_set, seed, data_size, n_clusters)

    # ========================================================================
    # Clustering methods
    if sys.argv[1] == 'kmeans':
        scores = []
        for n_clusters in range(4,25):
            # _, p1s, sps = load_data(inversion, data_set, seed, data_size, n_clusters)

            labels, score = kmeans_clustering(sps,
            dataset=data_set,
            inversion=inversion,
            n_clusters=n_clusters,
            pca_n_dim=pca_n_dim,
            tsne_seed=tsne_seed,
            tsne_size=tsne_size,
            plot=plot)

            scores.append(score)
            plot_cluster_result(inversion, p1s, labels, pca_n_dim, tsne_seed, L, rows=10)
            sys.exit()

        scores = np.array(scores)
        save_name = 'evaluation_scores_{}_{}_{}'.format(data_set, inversion, data_size)
        save_path = os.path.join(os.getcwd(),'clusters/kmeans',save_name)
        np.savetxt(save_path, scores, fmt='%.5f', delimiter=',')


    elif sys.argv[1] == 'chi_square':
        for inversion,data_size,cluster in best_clusters:
            data = np.zeros((14,1))
            for n_clusters in range(cluster,25):
                _, _, sps = load_data(inversion, data_set, seed, data_size, n_clusters)

                chi_square = chi_square_uniformity(sps,
                dataset=data_set,
                inversion=inversion,
                n_clusters=n_clusters,
                pca_n_dim=pca_n_dim)

                data[n_clusters-2,0] = chi_square
                break

            # save_name = 'chi_square_{}_{}'.format(data_set, data_size)
            # save_path = os.path.join(os.getcwd(),'clusters/chi_square',inversion,save_name)
            # np.savetxt(save_path, data, fmt='%.5f', delimiter=',')



if __name__ == '__main__':
    main()
    # generate_all_fourbar_mechanism_data()
