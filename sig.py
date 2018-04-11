from loader import *
from processData import *
from detect_head_shoulder import *
from display_patterns import *
from alexnet import *

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import mpl_finance #pip install https://github.com/matplotlib/mpl_finance/archive/master.zip
from matplotlib.dates import num2date
from tqdm import tqdm
import pickle
import os.path
import talib
import datetime


Yimg = 'img/datasetY.pkl'

width_img = 64
heigth_img = 48
nb_channels = 3

BATCH_SIZE = 128
NUM_CLASSES = 2
EPOCHS = 10

def split_data(X, Y, ratio=0.7):
    size = X.shape[0]
    X_train = X[:int(size*ratio)]
    X_test = X[int(size*ratio)-size:]
    Y_train = Y[:int(size*ratio)]
    Y_test = Y[int(size*ratio)-size:]

    return X_train, X_test, Y_train, Y_test

if __name__ == '__main__':
    print('Gathering data')
    #Run create_dataset first!
    #X: Reads all the images and places them in X,
    #Y: Reads the truth vector from the pickle created during create_dataset
    X, Y = load_data_from_imgs(Yimg)
    X_train, X_test, Y_train, Y_test = split_data(X, Y)
    print(X_train.shape)

    alexnet = get_alexNet((X_train.shape[1],X_train.shape[2],X_train.shape[3],), NUM_CLASSES)

    alexnet.summary()

    alexnet.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

    alexnet.fit(X_train, Y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        verbose=1,
                        validation_data=(X_test, Y_test))

    # Test the model
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
