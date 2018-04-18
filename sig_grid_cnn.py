from loader import *
from processData import *
from detect_head_shoulder import *
from display_patterns import *
from alexnet_grid import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpl_finance #pip install https://github.com/matplotlib/mpl_finance/archive/master.zip
from matplotlib.dates import num2date
from tqdm import tqdm
import pickle
import os.path
import talib
import datetime

from sklearn.metrics import confusion_matrix
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.optimizers import SGD, RMSprop
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from keras import backend as K

import h5py
import pydot
import graphviz

#img1 = high-low lines
#img2 = candlestick
#img3 = OHLC
#img4 = High
dir = 'img2/'
Yimg = dir + 'datasetY.pkl'
weights_file = 'best_alexnet.hdf5'

width_img = 64
heigth_img = 48
nb_channels = 3

BATCH_SIZE = 64
NUM_CLASSES = 2
EPOCHS = 50

history_log = []
model_log_files = []

def split_data(X, Y, ratio=0.7):
    size = X.shape[0]
    split_point = int(size*ratio)
    X_train = X[:split_point]
    X_test = X[split_point:]
    Y_train = Y[:split_point]
    Y_test = Y[split_point:]

    return X_train, X_test, Y_train, Y_test

if __name__ == '__main__':
    print('Gathering data')
    #Run create_dataset first!
    #X: Reads all the images and places them in X,
    #Y: Reads the truth vector from the pickle created during create_dataset
    X, Y = load_data_from_imgs(dir, Yimg)
    #X = np.concatenate((X, X), axis=0)
    #Y = np.concatenate((Y, Y), axis=0)
    X, Y = shuffle_in_unison(X, Y)
    X_train, X_test, Y_train, Y_test = split_data(X, Y, ratio=0.8)
    #X_test, X_val, Y_test, Y_val = split_data(X_test, Y_test, ratio=0.5)
    X_train = np.concatenate((X_train, X_train), axis=0)
    Y_train = np.concatenate((Y_train, Y_train), axis=0)

    print(len([y for y in Y_test if y[0]==1]))
    print(len([y for y in Y_test if y[1]==1]))
    labels = {0: len([y for y in Y if y[0]==1]), 1: len([y for y in Y if y[1]==1])}
    class_weight = create_class_weight(labels)
    print(class_weight)


    input_shape = (X_train.shape[1],X_train.shape[2],X_train.shape[3],)
    #alexnet = get_alexNet(input_shape, NUM_CLASSES, optimizers, init, dropout_rate)
    #alexnet = KerasClassifier(build_fn=get_alexNet_grid, verbose=1)

    dropout_rate = [0.7]
    lr=[0.003]
    #param_grid = dict(dropout_rate=dropout_rate, input_shape=input_shape, num_classes=num_classes, lr=lr)

    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=5, min_lr=0.000001, verbose=1)

    #grid = GridSearchCV(estimator=alexnet, param_grid=param_grid, fit_params={'callbacks': [checkpointer], 'validation_data': (X_val, Y_val), 'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'class_weight': class_weight})
    #grid_result = grid.fit(X, Y)


    #alexnet.summary()
    #plot_model(alexnet, to_file='model.png', show_shapes=True)

    #opt = SGD(lr=0.005, decay=1e-6, momentum=0.8, nesterov=True)
    #opt = RMSprop(lr=0.03, rho=0.8, epsilon=1e-05, decay=1e-5)





    for dropout in dropout_rate:
        for learning_rate in lr:
            fname =str(learning_rate)+str(dropout)+weights_file
            alexnet = get_alexNet_grid(input_shape=input_shape, num_classes=NUM_CLASSES, dropout_rate=dropout, lr=learning_rate)
            checkpointer = ModelCheckpoint(monitor='val_acc', filepath=fname, verbose=1, save_best_only=True, mode='max')
            datagen = ImageDataGenerator(horizontal_flip=True)
            '''history = alexnet.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch),
                                epochs=EPOCHS,
                                verbose=1,
                                callbacks=[checkpointer],
                                #class_weight = class_weight,
                                validation_data=(X_val, Y_val))'''
            history = alexnet.fit(X_train, Y_train, batch_size=BATCH_SIZE,
                                epochs=EPOCHS,
                                verbose=1,
                                callbacks=[checkpointer],
                                class_weight = class_weight,
                                validation_data=(X_test, Y_test))
            model_log_files.append(fname)
            history_log.append(history)
            K.clear_session()

    for history in history_log:
        plt.figure(1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='best')
        plt.show()

        plt.figure(2)
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='best')
        plt.show()
    '''history = alexnet.fit(X_train, Y_train, batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        verbose=1,
                        callbacks=[reduce_lr, checkpointer],
                        class_weight = class_weight,
                        validation_data=(X_test, Y_test))'''


    # summarize results
    '''print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
    	print("%f (%f) with: %r" % (mean, stdev, param))'''

    '''alexnet.load_weights(weights_file)
    # Test the model
    score = alexnet.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    pred = alexnet.predict(np.array(X_test))

    C = confusion_matrix([np.argmax(y) for y in Y_test], [np.argmax(y) for y in pred])

    print(C / C.astype(np.float).sum(axis=1))
    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    plt.show()

    plt.figure(2)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    plt.show()'''
