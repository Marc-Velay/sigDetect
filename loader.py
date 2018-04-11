from sig import Yimg, width_img, heigth_img, nb_channels
import csv
import pickle
import re
from datetime import timezone, datetime
from dateutil.parser import parse
from calendar import timegm
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import pandas as pd

from os import listdir
from os.path import isfile, join


def loadData(filename):

    data_original = pd.read_csv(filename)

    openp = data_original.ix[:, 'Open'].tolist()
    highp = data_original.ix[:, 'High'].tolist()
    lowp = data_original.ix[:, 'Low'].tolist()
    closep = data_original.ix[:, 'Close'].tolist()
    volumep = data_original.ix[:, 'Volume'].tolist()
    dates = data_original.ix[:, 'Local time'].tolist()

    data = np.column_stack((openp, highp, lowp, closep, volumep))

    return np.transpose(data)


def stringSplitByNumbers(x):
    r = re.compile('(\d+)')
    l = r.split(x)
    return [int(y) if y.isdigit() else y for y in l]


def load_data_from_imgs(y_file):
    files = [f for f in listdir('img/') if isfile(join('img/', f)) and '.pkl' not in f]
    if len(files) < 2:
        print('Run create_dataset.py first!')
        exit(1)
    files.sort(key=stringSplitByNumbers)
    X = np.empty((len(files), heigth_img, width_img, nb_channels))
    counter = 0

    with tqdm(total=len(files)) as pbar:
        for f in files:
            pbar.update(1)
            img = mpimg.imread('img/'+f)
            X[counter] = img[:, :, 0:3]

    with open(Yimg, 'rb') as fid:
            Y = pickle.load(fid)

    return X, Y
