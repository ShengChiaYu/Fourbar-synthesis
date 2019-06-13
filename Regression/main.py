import os
import sys
import numpy as np
import math
import matplotlib.pyplot as plt

from os.path import join
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

from util import parse_args, load_data_fourbar, predict

def svm_regression(x_train, x_test, label_train, label_test, data_param):
    # Fit regression model
    svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    svr_lin = SVR(kernel='linear', C=100, gamma='auto')
    svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,
               coef0=1)

    # Look at the results

    svrs = [
            svr_rbf,
            # svr_lin,
            # svr_poly
           ]
    kernel_label = [
                    'RBF',
                    # 'Linear',
                    # 'Polynomial',
                   ]

    for ix, svr in enumerate(svrs):
        print('kernel_label: {}\n'.format(kernel_label[ix]))

        svr.fit(x_train, label_train)
        pred_train = svr.predict(x_train)
        pred_test = svr.predict(x_test)

        train_acc = predict('{}_train'.format(data_param['target']), pred_train, label_train, write=False)
        test_acc = predict('{}_test'.format(data_param['target']), pred_test, label_test, write=False)

def main():
    args = parse_args()

    # Load fourbar data
    data_param = {'positions': args.pos_num, 'target': args.tar}
    x_train, x_test, label_train, label_test = load_data_fourbar(data_param)

    if args.svm:
        svm_regression(x_train, x_test, label_train, label_test, data_param)


if __name__ == '__main__':
    main()
