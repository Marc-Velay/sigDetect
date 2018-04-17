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
import random
import talib

#see http://projectfx.lusis:3000/fda/ai-bibliography/src/master/ebook/Deep_Learning_with_Python.pdf for data generator

load_from = '/home/mve/storage/data/Google_jan_mar_2017-8'

save_dir = 'img_large_OC/'
Yimg = save_dir + 'datasetY.pkl'

np.seterr(divide='ignore', invalid='ignore')


if __name__ == '__main__':
    print('Gathering data')
    data = loadData(load_from + '.csv')
    print('Creating training dataset')
    X, Y = createX_Y(data)


    len = len(X)
    X = np.transpose(X).reshape((5,len))
    WINDOW=25
    STEP=1
    Y = []
    counter = 0
    index = 0
    indexes=[]
    with tqdm(total=X.shape[1]-WINDOW) as pbar:
        for start in range(0, X.shape[1]-WINDOW, STEP):
            pbar.update(STEP)

            X2 = X[:, start:start+WINDOW]


            detected, indexes = detect_head_shoulder(X2)
            #res = talib.CDLIDENTICAL3CROWS(X2[0], X2[1], X2[2], X2[3])

            if not detected:
                #if detected == True:
                save_line_chart_inverted(X2, start, indexes, save_dir)
                Y.append([1., 0.])
                index+=1
            else:
                if counter%10 == 0:
                    save_line_chart_inverted(X2, start, indexes, save_dir)
                    Y.append([0., 1.])
                    index+=1
            counter+=1
    Y = np.array(Y)
    with open(Yimg, 'wb') as fid:
        pickle.dump(Y, fid)
