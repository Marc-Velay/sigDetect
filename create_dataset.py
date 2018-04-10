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
from tqdm import tqdm
import time

load_from = '/home/mve/storage/data/Google_jan_mar_2017-8'

Yname = 'yData.pkl'
Xname = 'xData.pkl'
Yimg = 'img/datasetY.pkl'

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

    len = len(X)
    X = np.transpose(X).reshape((5,len))
    WINDOW=25
    STEP=1
    Y = np.empty((X.shape[1]-WINDOW,2))
    counter = 0
    with tqdm(total=X.shape[1]-WINDOW) as pbar:
        for start in range(0, X.shape[1]-WINDOW, STEP):
            pbar.update(STEP)

            X2 = X[:, start:start+WINDOW]

            detected, indexes = detect_head_shoulder(X2)

            save_line_chart_inverted(X2, start)
            
            if detected == True:
                Y[counter] = [1., 0.]
            else:
                Y[counter] = [0., 1.]
            counter+=1

    with open(Yimg, 'wb') as fid:
        pickle.dump(Y, fid)
