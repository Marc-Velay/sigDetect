import numpy as np
import matplotlib.pyplot as plt
import mpl_finance #pip install https://github.com/matplotlib/mpl_finance/archive/master.zip
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas



def display_pattern_head_shoulder(X2, indexes):
    line_eqn = lambda x : ((X2[1][int(indexes[2])]-X2[1][int(indexes[0])])/(indexes[2]-indexes[0])) * (x - indexes[0]) + X2[1][int(indexes[0])]
    line_eqn_parr = lambda x : ((X2[1][int(indexes[2])]-X2[1][int(indexes[0])])/(indexes[2]-indexes[0])) * (x - indexes[0]) + X2[1][int(indexes[0])] - (line_eqn(indexes[1]) - X2[2][int(indexes[1])])

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
    #to have some space between the top of graph and window, for visibility
    margin = (max - min)/4
    ax.set_ylim([min - margin , max + margin])

    ohlc = np.column_stack((xdays, X2[0], X2[1], X2[2], X2[3]))

    horiz_line_top = np.array([X2[1][int(indexes[0])] for i in range(X2.shape[1])])
    plt.plot(horiz_line_top, 'r--', label='Top')
    plt.axvline(x=indexes[0], color='r', linewidth=0.3 )

    horiz_line_bot = np.array([X2[2][int(indexes[1])] for i in range(X2.shape[1])])
    plt.plot(horiz_line_bot, 'b--', label='Dip')
    plt.axvline(x=indexes[1], color='b', linewidth=0.3)

    horiz_line_next_top = np.array([X2[1][int(indexes[2])] for i in range(X2.shape[1])])
    plt.plot(horiz_line_next_top, 'y--', label='Second top')
    plt.axvline(x=indexes[2], color='y', linewidth=0.3)

    plt.plot([ line_eqn(x) for x in np.arange(X2.shape[1])], color='k', linestyle='-', linewidth=1)
    plt.plot([ line_eqn_parr(x) for x in np.arange(X2.shape[1])], color='k', linestyle='-', linewidth=1)


    mpl_finance.candlestick_ohlc(ax, ohlc, width=0.5, colorup='g', colordown='r')

    ax.autoscale_view()
    plt.show()

def save_line_chart_inverted(X, start):
    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.subplots(1)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    #ax.patch.set_facecolor('black')
    ax.axis('off')
    ax.plot(X[1], 'w')
    ax.plot(X[2], 'w')
    ax.axis('off')
    name = 'img/'+ str(start) +'.png'
    canvas.print_figure(name, dpi=10, frameon='false', facecolor='black')
    fig.clf()
    ax.cla()
    plt.close(fig)
    X = None
