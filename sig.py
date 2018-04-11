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

BATCH_SIZE = 128
num_classes = 2
epochs = 10

if __name__ == '__main__':
    print('Gathering data')
    #Run create_dataset first!
    #X: Reads all the images and places them in X,
    #Y: Reads the truth vector from the pickle created during create_dataset
    X, Y = load_data_from_imgs(Yimg)

    plt.plot(Y[:, 0])
    plt.show()

    alexnet = get_alexNet(X[:BATCH_SIZE].shape, num_classes)

    alexnet.summary()

    alexnet.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

    '''alexnet.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))
    '''
