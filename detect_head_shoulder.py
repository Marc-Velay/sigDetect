import numpy as np


def find_next_bottom(X):
    i = 0
    try:
        while i+1 <= X.shape[1]:
            close = is_close_to(X[2][i], X[2][i+1])
            if (X[2][i] >= X[2][i+1]):
                i+=1
            elif close == True:
                i+=1
            else:
                break
    except:
        pass
    return i


def find_next_top(X):
    i = 0
    try:
        while i+1 <= X.shape[1]:
            close = is_close_to(X[1][i], X[1][i+1])
            if (X[1][i] <= X[1][i+1]):
                i+=1
            elif close == True:
                i+=1
            else:
                break
    except:
        pass
    return i

def is_close_to(val1, val2):
    if abs(val1-val2) < 0:
        return True
    else:
        return False

def detect_head_shoulder(X2):

    indexes = np.zeros((3,1))
    # Get the peaks
    peaks = np.argpartition(X2[1], -2)[-2:]
    indexes[0] = peaks.min()

    #Indexes of the global minima and maxima succeding first minima
    indexes[1] = find_next_bottom(X2[:, int(indexes[0]):]) + indexes[0]
    indexes[2] = find_next_top(X2[:, int(indexes[1]):]) + indexes[1]

    dist_var_top_bot = np.linalg.norm(X2[1][int(indexes[0])]-X2[2][int(indexes[1])])
    dist_var_bot_top2 = np.linalg.norm(X2[2][int(indexes[1])]-X2[1][int(indexes[2])])
    #We compare the positions of the indexes, making sure they are in order min-max-min
    #The first minima should be smaller than the second
    #Check the ratios 
    if  indexes[0] < indexes[1]-1 and \
        indexes[1] < indexes[2]-1 and \
        X2[1][int(indexes[2])] < X2[1][int(indexes[0])] and \
        dist_var_bot_top2 > dist_var_top_bot*0.33 and \
        dist_var_bot_top2 < dist_var_top_bot*0.66 :
        return True, indexes
    else:
        return False, indexes
