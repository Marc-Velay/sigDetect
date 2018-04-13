from loader import *
from processData import *
from detect_head_shoulder import *
from display_patterns import *
from detect_double_top_bot import *

import numpy as np
import os.path
from tqdm import tqdm
import talib

load_from = '/home/mve/storage/data/Google_jan_mar_2017-8'


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
    with tqdm(total=X.shape[1]-WINDOW) as pbar:
        for start in range(0, X.shape[1]-WINDOW, STEP):
            pbar.update(STEP)

            X2 = X[:, start:start+WINDOW]

            res = talib.CDLIDENTICAL3CROWS(X2[0], X2[1], X2[2], X2[3])

            if not res.any() == 0:
                #print([i for i, j in enumerate(res) if not j == 0])
                #display_OHLC(X2)
                counter+=1
            '''trios = detect_double_top(X2)
            if trios:
                counter+=1
                #display_double_top_bot(X2, trios, True)'''
    print(counter)
