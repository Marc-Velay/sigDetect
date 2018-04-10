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

Ximg = 'img/xData_img.pkl'
Yimg = 'img/datasetY.pkl'

np.seterr(divide='ignore', invalid='ignore')


if __name__ == '__main__':
    print('Gathering data')
    X, Y = load_data_from_imgs(Yimg)

    print(X.shape)
    print(Y.shape)
