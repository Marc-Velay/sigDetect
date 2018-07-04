from loader import *
from processData import *
from detect_head_shoulder import *
from display_patterns import *
from d1_cnn import *

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
from sklearn.metrics import classification_report
from keras import backend as K

import h5py
import pydot
import graphviz

#img1 = high-low lines
#img2 = candlestick
#img3 = OHLC
#img4 = High
dir='1D-data/'
Yimg = dir + 'datasetY.pkl'
Ximg = dir + 'datasetX.pkl'
weights_file = dir + 'best_1dcnn.hdf5'

nb_channels = 3

BATCH_SIZE = 32
NUM_CLASSES = 2
EPOCHS = 100

history_log = []
model_log_files = []


if __name__ == '__main__':
    print('Gathering data')
    #Run create_dataset.py first!
    #X: Reads all the images and places them in X,
    #Y: Reads the truth vector from the pickle created during create_dataset
    random.seed(999)
    X, Y = load_data_from_pkl(dir, Yimg, Ximg)
    X, Y = subsample(X,Y)
    X, Y = shuffle_in_unison(X, Y)
    X_train, X_test, Y_train, Y_test = split_data(X, Y, ratio=0.8)

    '''print(len([y for y in Y_test if y==1]))
    print(len([y for y in Y_test if y==0]))
    labels = {0: len([y for y in Y if y==0]), 1: len([y for y in Y if y==1])}
    class_weight = create_class_weight(labels)
    print(class_weight)'''


    #input_shape = (X_train.shape[1],X_train.shape[2],X_train.shape[3],)

    dropout_rate = [0.4]
    lr=[0.01]
    #param_grid = dict(dropout_rate=dropout_rate, input_shape=input_shape, num_classes=num_classes, lr=lr)

    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=5, min_lr=0.000001, verbose=1)

    for dropout in dropout_rate:
        for learning_rate in lr:
            fname =weights_file #str(learning_rate)+str(dropout)+
            #1d_cnn = get_1dcnn_grid(input_shape=input_shape, num_classes=NUM_CLASSES, dropout_rate=dropout, lr=learning_rate)
            d1_cnn, optimizer = get_1dcnn_grid(nb_features=X_train.shape[1], nb_channels=X_train.shape[2], num_classes=NUM_CLASSES)
            checkpointer = ModelCheckpoint(monitor='val_acc', filepath=fname, verbose=1, save_best_only=True, mode='max')
            history = d1_cnn.fit(X_train, Y_train, batch_size=BATCH_SIZE,
                                epochs=EPOCHS,
                                verbose=1,
                                callbacks=[checkpointer],
                                validation_data=(X_test, Y_test))
            #model_log_files.append(fname)
            history_log.append(history)

            d1_cnn.load_weights(fname)
            # Test the model
            score = d1_cnn.evaluate(X_test[:BATCH_SIZE*math.floor(X_test.shape[0]/BATCH_SIZE)], Y_test[:BATCH_SIZE*math.floor(X_test.shape[0]/BATCH_SIZE)], verbose=0, batch_size=BATCH_SIZE)
            print('Test loss:', score[0])
            print('Test accuracy:', score[1])

            pred = d1_cnn.predict(np.array(X_test[:BATCH_SIZE*math.floor(X_test.shape[0]/BATCH_SIZE)]), batch_size=BATCH_SIZE)

            print(confusion_matrix([np.argmax(y) for y in Y_test[:BATCH_SIZE*math.floor(X_test.shape[0]/BATCH_SIZE)]], [np.argmax(y) for y in pred]))


            K.clear_session()
    '''
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
    '''

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
