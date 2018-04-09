from sig import *
import pickle

def data2change(data):
    change = pd.DataFrame(data).pct_change()
    change = change.replace([np.inf, -np.inf], np.nan)
    change = change.fillna(0.).values.tolist()
    change = [c[0] for c in change]
    return change

def remap(x, in_min, in_max, out_min, out_max):
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min



def createX_Y(data):
    X, Y = [], []
    print(len(data[0]))
    with tqdm(total=len(data[0])-1) as pbar:
        for i in range(0, len(data[0]-1), 1):
            pbar.update(1)
            try:
                o = np.array(data[0][i]).astype(float)
                h = np.array(data[1][i]).astype(float)
                l = np.array(data[2][i]).astype(float)
                c = np.array(data[3][i]).astype(float)
                v = np.array(data[4][i]).astype(float)
                '''
                o = remap(np.array(o), np.array(o).min(), np.array(o).max(), -1, 1)
                h = remap(np.array(h), np.array(h).min(), np.array(h).max(), -1, 1)
                l = remap(np.array(l), np.array(l).min(), np.array(l).max(), -1, 1)
                c = remap(np.array(c), np.array(c).min(), np.array(c).max(), -1, 1)
                v = remap(np.array(v), np.array(v).min(), np.array(v).max(), -1, 1)
                '''
                #print(o)

                x_i = np.column_stack((o, h, l, c, v))
                #x_i = x_i.flatten()
                y_i = data[3][i+1]
            except Exception as e:
                print(e)
                break

            if np.isnan(x_i).any() == False:
                X.append(x_i)
                Y.append(y_i)

    with open(Xname, 'wb') as fid:
        pickle.dump(X, fid)
    with open(Yname, 'wb') as fid:
        pickle.dump(Y, fid)
    print('using', len(X), 'minutes')
    return X, Y

def build_Y(data):
    new_Y = []


    return new_Y
