from loader import *
from processData import *
from detect_head_shoulder import *
from display_patterns import *

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import mpl_finance #Now a static file in current directory ##pip install https://github.com/matplotlib/mpl_finance/archive/master.zip
from matplotlib.dates import num2date
from tqdm import tqdm
import pickle
import os.path
from tqdm import tqdm
import time
import random
import talib

#see http://projectfx.lusis:3000/fda/ai-bibliography/src/master/ebook/Deep_Learning_with_Python.pdf for data generator

load_from = '/home/mve/storage/data/Google_jan_mar_2017-8'
#load_from = '/home/mve/storage/data/CAC40_jan_march_2018'
save_dir = '1D-data/'
Yimg = save_dir + 'datasetY.pkl'
Ximg = save_dir + 'datasetX.pkl'

np.seterr(divide='ignore', invalid='ignore')


if __name__ == '__main__':
    print('Gathering data')
    data = loadData(load_from + '.csv')
    print('Creating training dataset')
    X, Y = createX_Y(data)


    len = len(X)
    X = np.transpose(X).reshape((5,len)) #same index to access 5 lists OHLCV
    WINDOW=60
    STEP=1
    Y = []
    X_new = []
    counter = 0
    index = 0
    indexes=[]
    with tqdm(total=X.shape[1]-WINDOW) as pbar:
        for start in range(0, X.shape[1]-WINDOW, STEP):
            pbar.update(STEP)
            #sample WINDOW elements (15-30-60...) from the original dataset
            X2 = X[:, start:start+WINDOW]
            detected, indexes = detect_head_shoulder(X2)
            if detected:
                X_new.append(np.transpose(remap_ohlc(X2)))#remap_ohlc(X2)
                counter+=1
                Y.append([1., 0.])
            else:
                X_new.append(np.transpose(remap_ohlc(X2)))
                Y.append([0., 1.])
    print('Found: ' + str(counter) + ' patterns out of ' + str(X.shape[1]-WINDOW) + ' elements')
    Y = np.array(Y)
    X_new = np.array(X_new)
    print(X_new.shape)
    with open(Yimg, 'wb') as fid:
        pickle.dump(Y, fid)
    with open(Ximg, 'wb') as fid:
        pickle.dump(X_new, fid)
