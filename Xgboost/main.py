import os
import sys
import xgboost as xgb
import numpy as np
import math
import matplotlib.pyplot as plt

from os.path import join
from sklearn.preprocessing import MinMaxScaler

from util import parse_args, load_data


def load_data_fourbar(data_param):
    positions = data_param['positions']
    target = data_param['target']

    # label_column specifies the index of the column containing the true label
    x_train = load_data(join(os.getcwd(), '../Pytorch/data', positions, 'x_train.csv'), transform=True)
    y_train = load_data(join(os.getcwd(), '../Pytorch/data', positions, 'y_train_param.csv'), transform=False)
    x_test = load_data(join(os.getcwd(), '../Pytorch/data', positions, 'x_test.csv'), transform=True)
    y_test = load_data(join(os.getcwd(), '../Pytorch/data', positions, 'y_test_param.csv'), transform=False)
    print('Training data size:', x_train.shape)
    print('Testing data size:', x_test.shape)

    labels = {'r1':0, 'r3':1, 'r4':2, 'r5':3, 'th6':4}
    label_train = y_train[:, labels[target]]
    label_test = y_test[:, labels[target]]
    dtrain = xgb.DMatrix(data=x_train, label=label_train)
    dtest = xgb.DMatrix(data=x_test, label=label_test)
    print('Training target:', target)

    return dtrain, dtest, label_train, label_test


def predict(target, ypred, label, write=True):
    if write:
        save_path = join(os.getcwd(), 'predictions', '{}_pred.csv'.format(target))
        f = open(save_path, "w")
        count = 0
        threshold = 1e-2
        for i in range(ypred.shape[0]):
            diff_ratio = np.abs(ypred[i] - label[i]) / label[i]
            if diff_ratio <= threshold:
                count += 1
            f.write('{0:.4f}, {1:.4f}, {2:.4f}\n'.format(ypred[i], label[i], diff_ratio))
        acc = count/ypred.shape[0] * 100
        print('{0}_acc: {1:.3f}%'.format(target, acc))
        # f.write('{0}_acc: {1:.3f}%'.format(target, acc))
        f.close()

    else:
        count = 0
        threshold = 1e-2
        for i in range(ypred.shape[0]):
            diff_ratio = np.abs(ypred[i] - label[i]) / label[i]
            if diff_ratio <= threshold:
                count += 1
        acc = count/ypred.shape[0] * 100
        print('{0}_acc: {1:.3f}%'.format(target, acc))

    return acc


def train(args, dtrain, dtest, data_param, label_train, label_test):

    # Setting Parameters for booster
    param = {'max_depth': args.max_depth, 'eta': args.lr, 'verbosity': 1, 'objective': 'reg:linear'}
    param['eval_metric'] = 'mae'
    evallist = [(dtrain, 'train'), (dtest, 'eval')]

    # Training
    print('\nStart training...\n')
    num_round = args.max_epochs
    bst = xgb.train(param, dtrain, num_round, evals=evallist, early_stopping_rounds=args.early_stopping_rounds)

    # Predict
    ypred_train = bst.predict(dtrain)
    acc_train = predict('{}_train'.format(data_param['target']), ypred_train, label_train, write=False)
    ypred_test = bst.predict(dtest)
    acc_test = predict('{}_test'.format(data_param['target']), ypred_test, label_test, write=False)

    # Save model
    save_path = join(os.getcwd(), args.save_dir, '{}.model'.format(data_param['target']))
    bst.save_model(save_path)
    print('\nSave model to {}\n'.format(save_path))


def test_sine():
    np.random.seed(1124)
    X_train = np.sort(np.random.rand(100) * 2 * math.pi)
    X_test = np.sort(np.random.rand(100) * 2 * math.pi)
    Y_train = np.sin(X_train)
    Y_test = np.sin(X_test)
    dtrain = xgb.DMatrix(data=np.expand_dims(X_train, axis=1), label=Y_train)
    dtest = xgb.DMatrix(data=np.expand_dims(X_test, axis=1), label=Y_test)

    # Setting Parameters for booster
    param = {'max_depth': 2, 'eta': 1, 'verbosity': 1, 'objective': 'reg:linear'}
    param['eval_metric'] = 'mae'
    evallist = [(dtest, 'eval'), (dtrain, 'train')]

    # Training
    num_round = 100
    bst = xgb.train(param, dtrain, num_round, evallist)

    # Save model
    bst.save_model(join(os.getcwd(), 'test_sine', 'sin.model'))

    # Predict
    ypred_train = bst.predict(dtrain)
    predict('sin_train', ypred_train, Y_train)
    ypred_test = bst.predict(dtest)
    predict('sin_test', ypred_test, Y_test)

    # Plot
    plt.plot(X_train, ypred_train, label='pred')
    plt.plot(X_train, Y_train, label='gt')
    plt.title('Sine_train')
    plt.legend()
    save_path = join(os.getcwd(), 'test_sine', 'sine_train.jpg')
    plt.savefig(save_path, bbox_inches='tight')

    plt.plot(X_test, ypred_test, label='pred')
    plt.plot(X_test, Y_test, label='gt')
    plt.title('Sine_test')
    plt.legend()
    save_path = join(os.getcwd(), 'test_sine', 'sine_test.jpg')
    plt.savefig(save_path, bbox_inches='tight')


def main():
    args = parse_args()

    # Load fourbar data
    data_param = {'positions': args.pos_num, 'target': args.tar}
    dtrain, dtest, label_train, label_test = load_data_fourbar(data_param)

    if args.load_model:
        # Load model
        bst = xgb.Booster({'nthread': 4})  # init model
        load_path = join(os.getcwd(), args.save_dir, '{}.model'.format(data_param['target']))
        bst.load_model(load_path)  # load data
        print('\nLoad model...{}\n'.format(load_path))

        # Predict
        ypred_train = bst.predict(dtrain)
        acc_train = predict('{}_train'.format(data_param['target']), ypred_train, label_train)
        ypred_test = bst.predict(dtest)
        acc_test = predict('{}_test'.format(data_param['target']), ypred_test, label_test)

    elif args.tune:
        save_path = join(os.getcwd(), 'accuracy', '{}_accuracy.csv'.format(data_param['target']))
        f = open(save_path, "w")
        f.write('Training target: {}\n'.format(data_param['target']))
        f.write('max_depth, eta, acc_train, acc_test\n')
        acc_best = 0
        for i in range(10):
            for j in range(10):
                # Setting Parameters for booster
                param = {'max_depth': 6+i, 'eta': 1/(j+1), 'verbosity': 1, 'objective': 'reg:linear'}
                param['eval_metric'] = 'mae'
                evallist = [(dtrain, 'train'), (dtest, 'eval')]

                # Training
                print('\nStart training...\n')
                num_round = args.max_epochs
                bst = xgb.train(param, dtrain, num_round, evals=evallist, early_stopping_rounds=args.early_stopping_rounds)

                # Predict
                ypred_train = bst.predict(dtrain)
                acc_train = predict('{}_train'.format(data_param['target']), ypred_train, label_train, write=False)
                ypred_test = bst.predict(dtest)
                acc_test = predict('{}_test'.format(data_param['target']), ypred_test, label_test, write=False)

                if acc_test > acc_best:
                    acc_best = acc_test
                    f.write('{0}, {1:.3f}, {2:.3f}, {3:.3f}'.format(param['max_depth'], param['eta'], acc_train, acc_test))
                    f.write(' --> best\n')

                    # Save model
                    save_path = join(os.getcwd(), args.save_dir, '{}.model'.format(data_param['target']))
                    bst.save_model(save_path)
                    print('\nSave model to {}\n'.format(save_path))

                else:
                    f.write('{0}, {1:.3f}, {2:.3f}, {3:.3f}\n'.format(param['max_depth'], param['eta'], acc_train, acc_test))
                f.flush()
    else:
        train(args, dtrain, dtest, data_param, label_train, label_test)

if __name__ == '__main__':
    main()
