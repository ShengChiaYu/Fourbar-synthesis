import numpy as np
np.random.seed(1124)
import matplotlib.pyplot as plt
import sys, os
import math
import csv
from cmath import sqrt
from math import cos, sin, atan, acos, pi, degrees
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.datasets import mnist
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Generate path by parameters of fourbar linkages
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
        plt.title("Upper limit exists.")
        plt.xlabel("Th_2 min = {:.3f}, Th_2 max = {:.3f}".format(degrees(-th2_max.real), degrees(th2_max.real)))
        print("Upper limit exists.")
    # Lower limit exists
    elif condition_1 <= 0 and condition_2 < 0:
        th2_min = acos((L[0]**2 + L[1]**2 - (L[2] - L[3])**2) / (2*L[0]*L[1]))
        th2 = np.linspace(th2_min, 2*pi - th2_min, n)
        plt.title("Lower limit exists.")
        plt.xlabel("Th_2 min = {:.3f}, Th_2 max = {:.3f}".format(degrees(th2_min.real), degrees(2*pi-th2_min.real)))
        print("Lower limit exists.")
    # Both limit exist
    elif condition_1 > 0 and condition_2 < 0:
        th2_max = acos((L[0]**2 + L[1]**2 - (L[2] + L[3])**2) / (2*L[0]*L[1]))
        th2_min = acos((L[0]**2 + L[1]**2 - (L[2] - L[3])**2) / (2*L[0]*L[1]))
        th2 = np.linspace(th2_min, th2_max, n)
        #th2 = np.linspace(-th2_max, -th2_min, n)
        plt.title("Both limit exist.")
        plt.xlabel("Th_2 min = {:.3f}, Th_2 max = {:.3f}".format(degrees(th2_min.real), degrees(th2_max.real)))
        print("Both limit exist.")
    # No limit exists
    elif condition_1 <= 0 and condition_2 >= 0:
        th2 = np.linspace(0, 2*pi, n)
        plt.title("No limit exists.")
        plt.xlabel("Th_2 min = 0, Th_2 max = 360")
        print("No limit exists.")

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

        plt.plot(L[1]*cos(th2[i]) + x0, L[1]*sin(th2[i]) + y0, 'go', markersize=1)

    p1 = np.array(p1)
    p2 = np.array(p2)

    plt.plot(p1[:,0],p1[:,1],'ro', markersize=1)
    plt.plot(p2[:,0],p2[:,1],'bo', markersize=1)
    plt.plot(x0,y0,'k+')
    plt.axis('equal')
    plt.show()

    return p1, p2

# Read datasets
def load_data(file):
    raw_data = np.genfromtxt(file, delimiter=',')
    scaler = MinMaxScaler(copy=True, feature_range=(-1, 1))
    data = scaler.fit_transform(raw_data)
    return data, scaler

# Dnn trianing model
def train(x_train, y_train, x_test, y_test, batch, epochs, dpr, patience, model_num):
    model = models(model_num, dpr)

    save_path = os.path.join(os.getcwd(), 'models', data_folder, '{}_batch{}_dpr{}.h5'.format(model_num, batch, dpr))
    callbacks = [EarlyStopping(monitor='val_loss', patience=patience),
                 ModelCheckpoint(filepath=save_path, monitor='val_loss', save_best_only=True)]

    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]

    history = model.fit(x_train, y_train, batch_size=batch, epochs=epochs, callbacks=callbacks,
                        validation_split=0.2)

    # Save evaluations to loss_history.
    score = model.evaluate(x_train, y_train)
    result = model.evaluate(x_test, y_test)
    save_path = os.path.join(os.getcwd(), 'loss_history.csv')
    f_loss = open(save_path, "a+")
    f_loss.write('nodel: {}, batch: {}, dpr: {}\n'.format(model_num, batch, dpr))
    f_loss.write('Train loss: {0:.8f}, Test loss: {1:.8f}\n\n'.format(score[0], result[0]))
    f_loss.close()

    # plot training history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='valid')
    plt.title('batch: {}, dpr: {}'.format(batch, dpr) )
    plt.legend()
    save_path = os.path.join(os.getcwd(), 'training_history', data_folder, '{}_batch{}_dpr{}.png'.format(model_num, batch, dpr))
    plt.savefig(save_path, bbox_inches='tight')

    return [model, batch, dpr]

# Predict test data by training model
def predict(model, x_train, y_train, x_test, y_test, scaler, model_num):

    # Print evaluation.
    score = model[0].evaluate(x_train, y_train)
    print("\nTrain loss:", score[0])
    result = model[0].evaluate(x_test, y_test)
    print("\nTest loss:", result[0])

    # Write file
    answer = scaler.inverse_transform(model[0].predict(x_test))
    save_path = os.path.join(os.getcwd(), 'results', data_folder, '{}_batch{}_dpr{}.csv'.format(model_num, model[1], int(10*model[2]) ) )
    f_result = open(save_path, "w")
    for i in range(x_test.shape[0]):
        content = " ".join(str(x) for x in answer[i,:])
        f_result.write(content)
        f_result.write('\n')
    f_result.close()

    return answer

# Print evaluations
def print_eval(model, x_train, y_train, x_test, y_test, scaler, model_num):
    # Print evaluation.
    score = model[0].evaluate(x_train, y_train)
    print("\nTrain loss:", score[0])
    result = model[0].evaluate(x_test, y_test)
    print("\nTest loss:", result[0])

# Models
def models(n,dpr):
    # Paper model
    if n == 1:
        model = Sequential()
        model.add(Dense(input_dim=22,units=22,activation='tanh'))
        model.add(Dropout(dpr))
        model.add(Dense(units=5,activation='tanh'))
        model.add(Dropout(dpr))
        model.add(Dense(units=5,activation='tanh'))
        model.add(Dropout(dpr))
        model.add(Dense(units=5,activation='linear'))

        model.compile(loss='mse',optimizer='adam',metrics=['mse'])

    elif n == 2:
        model = Sequential()
        model.add(Dense(input_dim=22,units=32,activation='tanh'))
        model.add(Dropout(dpr))
        model.add(Dense(units=64,activation='tanh'))
        model.add(Dropout(dpr))
        model.add(Dense(units=32,activation='tanh'))
        model.add(Dropout(dpr))
        model.add(Dense(units=5,activation='linear'))

        model.compile(loss='mse',optimizer='adam',metrics=['mse'])


    return model

# Main()
if __name__ == "__main__":

    # Testing data
    # L = np.array([30, 40, 31, 38]) # Upper limit exists. c1>0  c2>=0
    # L = np.array([20, 29, 30, 40]) # Lower limit exists. c1<=0 c2<0
    # L = np.array([31, 40, 20, 40]) # Both limit exist.   c1>0  c2<0
    # L = np.array([20, 40, 30, 40]) # No limit exists.    c1<=0 c2>=0
    # th1 = 0
    # r = L[2]/2*math.sqrt(2) # 50 % of coupler length
    # alpha = pi/4       # midpoint of coupler link
    # n = 360
    # x0 = 0
    # y0 = 0

    # Q.J.Ge Closed path parameters
    # L = np.array([11, 6, 8, 10])
    # th1 = 0.1745
    # r = 7
    # alpha = 0.6981
    # n = 360
    # x0 = 10
    # y0 = 14

    # Q.J.Ge open path parameters
    # L = np.array([3, 1, 2, 1.6])
    # th1 = 0.2
    # r = 0.5
    # alpha = 0.3
    # n = 360
    # x0 = -2
    # y0 = -3

    # =============== Path generation ===============
    #p1, p2 = path_gen_open(L, th1, r, alpha, n, x0, y0)

    # ================== Training ===================
    data_folder = 'threshold1e-3'
    x_train = load_data(os.path.join(os.getcwd(), 'data', data_folder, 'x_train.csv')) # load_data() return (data, scaler)
    y_train = load_data(os.path.join(os.getcwd(), 'data', data_folder, 'y_train.csv'))
    x_test = load_data(os.path.join(os.getcwd(), 'data', data_folder, 'x_test.csv'))
    y_test = load_data(os.path.join(os.getcwd(), 'data', data_folder, 'y_test.csv'))
    model_num = 2

    # model = train(x_train[0], y_train[0], x_test[0], y_test[0],
    #               batch=64, epochs=1000, dpr=0.2, patience=10, model_num=model_num) # train() return eturn (model, batch, dpr)
    # result = predict(model, x_train[0], y_train[0], x_test[0], y_test[0], y_train[1], model_num=model_num)

    # ============ Load models and print evaluation =============
    model = load_model(os.path.join(os.getcwd(), 'models', data_folder, '2_batch64_dpr0.2.h5'))
    model = [model, 64, 0.2]
    print_eval(model, x_train[0], y_train[0], x_test[0], y_test[0], y_train[1], model_num)
