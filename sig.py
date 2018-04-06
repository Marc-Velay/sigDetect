from loader import *
from processData import *

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
#from matplotlib.finance import candlestick_ohlc
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

def find_next_bottom(X):
    max_bot = X[2][0]
    i = 0
    try:
        while i+1 <= X.shape[1]:
            if X[2][i] > X[2][i+1]:
                i+=1
            else:
                break
    except:
        pass
    return i


def find_next_top(X):
    min_bot = X[1][0]
    i = 0
    try:
        while i+1 <= X.shape[1]:
            if X[1][i] < X[1][i+1]:
                i+=1
            else:
                break
    except:
        pass
    return i


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
    STEP=20
    for start in range(0, X.shape[1], 2):#STEP):
        print(start)
        X2 = X[:, start:start+STEP]
        Y2 = Y[start:start+STEP]
        #indicators = talib.CDLTRISTAR(X[0], X[1], X[2], X[3])
        #plt.plot(indicators)
        #plt.show()


        # Get the peaks
        peaks = np.argpartition(X2[1], -2)[-2:]
        top_value = X2[1][peaks.min()]
        top_index = peaks.min()

        bot_index = find_next_bottom(X2[:, top_index:]) + top_index
        second_top_index = find_next_top(X2[:, bot_index:]) + bot_index

        dist_var_top_bot = np.linalg.norm(X2[1][top_index]-X2[2][bot_index])
        dist_var_bot_top2 = np.linalg.norm(X2[2][bot_index]-X2[1][second_top_index])


        if  top_index < bot_index-1 and \
            bot_index < second_top_index-1 and \
            X2[1][second_top_index] < X2[1][top_index] and \
            dist_var_bot_top2 > dist_var_top_bot*0.33 and dist_var_bot_top2 < dist_var_top_bot*0.66:
            line_eqn = lambda x : ((X2[1][second_top_index]-X2[1][top_index])/(second_top_index-top_index)) * (x - top_index) + top_value
            line_eqn_parr = lambda x : ((X2[1][second_top_index]-X2[1][top_index])/(second_top_index-top_index)) * (x - top_index) + top_value - (line_eqn(bot_index) - X2[2][bot_index])


            # determine number of days and create a list of those days
            ndays = np.unique(np.trunc(X2[0,:]), return_index=True)
            xdays =  np.arange(X2.shape[1]) #assume that processData has returned a uniform array

            # plot the data
            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_axes([0.1, 0.2, 0.85, 0.7])
            # customization of the axis
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            ax.tick_params(axis='both', direction='out', width=2, length=8,
                           labelsize=12, pad=8)
            ax.spines['left'].set_linewidth(2)
            ax.spines['bottom'].set_linewidth(2)

            # set the ticks of the x axis only when starting a new day
            ax.set_xticks(xdays)
            ax.set_xticklabels(xdays, rotation=45, horizontalalignment='right')

            ax.set_ylabel('Value (USD)', size=20)
            max = X2[1].max()
            min = X2[2].min()
            margin = (max - min)/4
            ax.set_ylim([min - margin , max + margin])

            ohlc = np.column_stack((xdays, X2[0], X2[1], X2[2], X2[3]))

            horiz_line_top = np.array([top_value for i in range(X2.shape[1])])
            plt.plot(horiz_line_top, 'r--', label='Top')
            plt.axvline(x=peaks.min(), color='r', linewidth=0.3 )

            horiz_line_bot = np.array([X2[2][bot_index] for i in range(X2.shape[1])])
            plt.plot(horiz_line_bot, 'b--', label='Dip')
            plt.axvline(x=bot_index, color='b', linewidth=0.3)
            horiz_line_next_top = np.array([X2[1][second_top_index] for i in range(X2.shape[1])])
            plt.plot(horiz_line_next_top, 'y--', label='Second top')
            plt.axvline(x=second_top_index, color='y', linewidth=0.3)

            plt.plot([ line_eqn(x) for x in np.arange(X2.shape[1])], color='k', linestyle='-', linewidth=1)
            plt.plot([ line_eqn_parr(x) for x in np.arange(X2.shape[1])], color='k', linestyle='-', linewidth=1)


            '''legend = ax.legend(loc='upper center', shadow=False)
            frame = legend.get_frame()
            frame.set_facecolor('0.90')

            # Set the fontsize
            for label in legend.get_texts():
                label.set_fontsize('large')
            for label in legend.get_lines():
                label.set_linewidth(1.5)  # the legend line width'''

            mpl_finance.candlestick_ohlc(ax, ohlc, width=0.5, colorup='g', colordown='r')

            ax.autoscale_view()
            plt.show()
