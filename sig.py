from loader import *
from processData import *
from detect_head_shoulder import *
from display_patterns import *

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


load_from = '/home/mve/storage/data/Google_jan_mar_2017-8'

Yname = 'yData.pkl'
Xname = 'xData.pkl'

np.seterr(divide='ignore', invalid='ignore')


if __name__ == '__main__':
    if not os.path.isfile(Xname):
        print('Gathering data')
        data = loadData(load_from + '.csv')
        print('Creating training dataset')
        X, Y = createX_Y(data)

    else:
        with open(Xname, 'rb') as fid:
                X = pickle.load(fid)
        with open(Yname, 'rb') as fid:
                Y = pickle.load(fid)

    len = len(X)
    X = np.transpose(X).reshape((5,len))
    WINDOW=25
    STEP=1
    counter = 0
    for start in range(0, X.shape[1]-WINDOW, STEP):#STEP):
        #print(start)
        X2 = X[:, start:start+WINDOW]
        Y2 = Y[start:start+WINDOW]
        #indicators = talib.CDLTRISTAR(X[0], X[1], X[2], X[3])
        #plt.plot(indicators)
        #plt.show()

        detected, indexes = detect_head_shoulder(X2)


        if detected == True:
            #display_pattern_head_shoulder(X2, indexes)
            counter +=1
    print(counter)
    print(X.shape[1])
    print(X.shape[1]/WINDOW)
