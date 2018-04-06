import csv
from datetime import timezone, datetime
from dateutil.parser import parse
from calendar import timegm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns
sns.despine()

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
