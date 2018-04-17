from loader import *
from processData import *
from detect_head_shoulder import *
from display_patterns import *
from alexnet import *

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
import h5py
import pydot
import graphviz

#img1 = high-low lines
#img2 = candlestick
#img3 = OHLC
#img4 = High
dir = 'img4/'
Yimg = dir + 'datasetY.pkl'
weights_file = 'best_alexnet.hdf5'

width_img = 64
heigth_img = 48
nb_channels = 3

BATCH_SIZE = 128
NUM_CLASSES = 2
EPOCHS = 50
seed = 128

def split_data(X, Y, ratio=0.7):
    size = X.shape[0]
    split_point = int(size*ratio)
    X_train = X[:split_point]
    X_test = X[split_point:]
    Y_train = Y[:split_point]
    Y_test = Y[split_point:]

    return X_train, X_test, Y_train, Y_test

if __name__ == '__main__':
    rng = np.random.RandomState(seed)
    print('Gathering data')
    #Run create_dataset first!
    #X: Reads all the images and places them in X,
    #Y: Reads the truth vector from the pickle created during create_dataset
    X, Y = load_data_from_imgs(dir, Yimg)
    #X, Y = shuffle_in_unison(X, Y)
    X_train, X_test, Y_train, Y_test = split_data(X, Y, ratio=0.6)
    X_test, X_val, Y_test, Y_val = split_data(X_test, Y_test, ratio=0.5)

    print(len([y for y in Y_test if y[0]==1]))
    print(len([y for y in Y_test if y[1]==1]))
    labels = {0: len([y for y in Y if y[0]==1]), 1: len([y for y in Y if y[1]==1])}
    class_weight = create_class_weight(labels)
    print(class_weight)
    alexnet = get_alexNet((X_train.shape[1],X_train.shape[2],X_train.shape[3],), NUM_CLASSES)

    #alexnet.summary()
    #plot_model(alexnet, to_file='model.png', show_shapes=True)

    opt = SGD(lr=0.005, decay=1e-6, momentum=0.8, nesterov=True)
    #opt = RMSprop(lr=0.03, rho=0.8, epsilon=1e-05, decay=1e-5)

    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=5, min_lr=0.000001, verbose=1)
    checkpointer = ModelCheckpoint(monitor='val_acc', filepath=weights_file, verbose=1, save_best_only=True, mode='max')

    alexnet.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

    datagen = ImageDataGenerator(horizontal_flip=True)

    '''history = alexnet.fit_generator(datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE),
                        epochs=EPOCHS,
                        verbose=1,
                        callbacks=[reduce_lr, checkpointer],
                        class_weight = class_weight,
                        validation_data=(X_test, Y_test))'''

    history = alexnet.fit(X_train, Y_train, batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        verbose=1,
                        callbacks=[reduce_lr, checkpointer],
                        class_weight = class_weight,
                        validation_data=(X_val, Y_val))


    alexnet.load_weights(weights_file)
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
    plt.show()
