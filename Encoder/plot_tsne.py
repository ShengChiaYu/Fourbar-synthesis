import numpy as np
np.random.seed(1216)
import matplotlib.pyplot as plt
import pandas as pd

import glob
import os
import sys
import argparse
from os.path import join

from PIL import Image
from tqdm import tqdm
from sklearn.manifold import TSNE


def read_features(file_name):
    print('\nLoad features...')
    file_path = join(os.getcwd(), 'features', file_name)
    my_data = np.genfromtxt(file_path, delimiter=',')
    image_index = my_data[:,0]
    features = my_data[:,1:-1]
    print('Done')

    return image_index, features


def plot(image_index, features):
    tsne = TSNE(n_components=2, random_state=0)
    # sample 5000 features to do tsne
    sample_ind = np.random.randint(features.shape[0], size=5000)
    sample_features = features[sample_ind]

    print('\nTsne transform...')
    tsne_features = tsne.fit_transform(sample_features)
    print('Save tsne_features.npy')
    np.save('features/tsne_features.npy',tsne_features)

    ## random pick 20 images to visulize the distribution
    print('Save tsne_features.csv')
    set_size = features.shape[0]/3

    rand_img = np.random.randint(sample_features.shape[0], size=20)
    coord = tsne_features[rand_img]
    # index of the 20 images in the 5000 samples
    rand_img_ind = sample_ind[rand_img]
    set = np.floor(rand_img_ind/set_size)+1
    img_index = image_index[rand_img_ind]

    df = pd.DataFrame({'Set': set,
                       'Index': img_index,
                       'X': coord[:,0],
                       'Y': coord[:,1],
                      })

    df = df[['Set', 'Index', 'X', 'Y']]
    df.to_csv('features/tsne_features.csv')
    print('Finish!')


def main():
    feature_filename = 'features_GCRR_autoencoder.txt'

    image_index, features = read_features(feature_filename)
    plot(image_index, features)



if __name__ == '__main__':
    main()
