from loader import *
from processData import *
#from detect_head_shoulder import *
#from display_patterns import *
from lstm_grid import get_lstm_grid

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import os.path
import talib
import math

from sklearn.metrics import confusion_matrix
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
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
weights_file = dir + 'best_lstm.hdf5'

WINDOW = 30
STEP = 1
NUM_CLASSES = 2
EPOCHS = 50
BATCH_SIZE = 64

history_log = []
scores = []

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
    data = loadData(load_from + '.csv')
    print('Creating training dataset')
    X, nop = createX_Y(data)
    len = len(X)
    X = np.transpose(X).reshape((5,len))
    X, Y = createX_Y_frames(X, WINDOW, STEP)
    X, Y = shuffle_in_unison(X, Y)
    #X, Y = rebalancing(X, Y)
    '''X, Y = load_data_from_pkl(dir, Yimg, Ximg)
    X, Y = subsample(X,Y)
    X, Y = shuffle_in_unison(X, Y)'''

    #print(X.shape, Y.shape)

    X_train, X_test, Y_train, Y_test = split_data(X, Y, ratio=0.8)
    #X_test, X_val, Y_test, Y_val = split_data(X_test, Y_test, ratio=0.5)


    #print(Y_test[:,0].sum())

    #print((Y[:,1].sum()/(Y[:,1].sum()+Y[:,0].sum()))*100)
    labels = {0: Y[:,0].sum(), 1: Y[:,1].sum()}
    class_weight = create_class_weight(labels)


    input_shape = (BATCH_SIZE,X_train.shape[1], X_train.shape[2])

    dropout=[0.5]
    learning_rate=[0.00006]#np.arange(0.00001, 0.0001, 0.00001) #[0.00006]

    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=5, min_lr=0.000001, verbose=0)

    for lr in learning_rate:
        for dout in dropout:
            fname =weights_file #str(dout)+str(lr)+
            lstm_model = get_lstm_grid(input_shape=input_shape, num_classes=NUM_CLASSES, lr=lr, dropout=dout)
            checkpointer = ModelCheckpoint(monitor='val_acc', filepath=fname, verbose=0, save_best_only=True, mode='max')
            '''datagen = ImageDataGenerator(horizontal_flip=True)
            history = alexnet.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch),
                                epochs=EPOCHS,
                                verbose=1,lstm_model.load_weights(fname)
                                callbacks=[checkpointer],
                                #class_weight = class_weight,
                                validation_data=(X_val, Y_val))'''
            history = lstm_model.fit(X_train[:BATCH_SIZE*math.floor(X_train.shape[0]/BATCH_SIZE)], Y_train[:BATCH_SIZE*math.floor(X_train.shape[0]/BATCH_SIZE)], batch_size=BATCH_SIZE,
                                epochs=EPOCHS,
                                verbose=1,
                                callbacks=[reduce_lr, checkpointer],
                                #class_weight = class_weight,
                                validation_data=(X_test[:BATCH_SIZE*math.floor(X_test.shape[0]/BATCH_SIZE)], Y_test[:BATCH_SIZE*math.floor(X_test.shape[0]/BATCH_SIZE)]))
            history_log.append(history)

            lstm_model.load_weights(fname)
            # Test the model
            score = lstm_model.evaluate(X_test[:BATCH_SIZE*math.floor(X_test.shape[0]/BATCH_SIZE)], Y_test[:BATCH_SIZE*math.floor(X_test.shape[0]/BATCH_SIZE)], verbose=0, batch_size=BATCH_SIZE)
            print('Test loss:', score[0])
            print('Test accuracy:', score[1])
            print(fname)

            pred = lstm_model.predict(np.array(X_test[:BATCH_SIZE*math.floor(X_test.shape[0]/BATCH_SIZE)]), batch_size=BATCH_SIZE)

            print(confusion_matrix([np.argmax(y) for y in Y_test[:BATCH_SIZE*math.floor(X_test.shape[0]/BATCH_SIZE)]], [np.argmax(y) for y in pred]))


    '''for history in history_log:
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

    input_shape = (1,X_test.shape[1], X_test.shape[2])
    for lr in learning_rate:
        for dout in dropout:
            for example_index in range(X_test.shape[0]):
                fname =weights_file #str(dout)+str(lr)+
                lstm_model = get_lstm_grid(input_shape=input_shape, num_classes=NUM_CLASSES, lr=lr, dropout=dout)
                lstm_model.load_weights(fname)
                pred = lstm_model.predict(X_test[example_index].reshape(1,X_test[example_index].shape[0],X_test[example_index].shape[1]), batch_size=1)
                print('predicted: ', pred, ' was: ', Y_test[example_index])

                X2 = X_test[example_index].transpose()
                detected, indexes = detect_head_shoulder(X2)
                #if detected:
                display_pattern_head_shoulder(X2, indexes)

    K.clear_session()
