import numpy as np
from scipy.signal import argrelextrema
from display_patterns import *



def find_doubles(X, maximas):
    MARGIN_TOPS = PIP
    pairs = []
    for array in maximas:
        nb_maximas = array.shape[0]
        if nb_maximas > 1:
            for m1 in range(0, nb_maximas):
                for m2 in range(nb_maximas, 0, -1):
                    if not m1 == m2:
                        diff = abs(X[m1]-X[m2])
                        if diff <= MARGIN_TOPS:
                            pairs.append([m1, m2])
    return pairs

def find_trios(couples, maximas):
    trios = []

    for pair in couples:
        for maxima in maximas[0]:
            if pair[0] < maxima and maxima < pair[1]:
                trios.append([pair, maxima])
    return trios

def filter_trios(X, trios):
    MARGIN_TOP_BOT = 50*PIP
    new_trios = []
    is_max = True
    is_min = True
    for trio in trios:
        dip = abs(X[1][trio[0][0]]-X[2][trio[1]])
        for i in range(trio[0][0]+1, trio[0][1]):
            if X[1][i] >= X[1][trio[0][0]] or X[1][i] >= X[1][trio[0][1]]:
                is_max = False
                break
            if X[2][i] <= X[1][trio[1]] and not i==trio[1]:
                is_min = False
                break
                #and is_max == True and is_min == True
        if dip > MARGIN_TOP_BOT and is_max == True and is_min == True:
            new_trios.append(trio)
    return new_trios

def detect_double_top(X):
    global PIP
    PIP = (X[1].max())/10000
    maxOC = np.vstack([X[0], X[3]]).max(axis=0)
    minOC = np.vstack([X[0], X[3]]).min(axis=0)
    maxs = argrelextrema(maxOC, np.greater)
    mins = argrelextrema(minOC, np.less)

    if max:
        couples = find_doubles(X[1], maxs)
        if couples and mins:
            trios = find_trios(couples, mins)
            if trios:
                trios = filter_trios(X, trios)
                if trios:
                    print(trios)
                    display_double_top_bot(X, trios, maxs, mins, True)
                    return trios
    return []
