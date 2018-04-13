from sig import *
from create_dataset import *
import pickle
from collections import Counter
import math

def data2change(data):
    change = pd.DataFrame(data).pct_change()
    change = change.replace([np.inf, -np.inf], np.nan)
    change = change.fillna(0.).values.tolist()
    change = [c[0] for c in change]
    return change

def remap(x, in_min, in_max, out_min, out_max):
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def rebalancing(x_train, y_train):

    ## Up/Downsample Mid Class of Categorical 3 Classes
    from sklearn.utils import resample

    class_indexes0 = np.where(y_train[:,0] == 1)[0]
    class_indexes1 = np.where(y_train[:,1] == 1)[0]
    class_indexes2 = np.where(y_train[:,2] == 1)[0]

    n_samples=min(len(class_indexes0),len(class_indexes2))

    y_resampled1 = resample(class_indexes1, replace=True,n_samples=n_samples,random_state=1000)

    x_train = np.concatenate((x_train[class_indexes0],x_train[y_resampled1],x_train[class_indexes2]))
    y_train = np.concatenate((y_train[class_indexes0],y_train[y_resampled1],y_train[class_indexes2]))

    train_indexes = np.arange(x_train.shape[0])

    np.random.shuffle(train_indexes)

    x_train = x_train[train_indexes]
    y_train = y_train[train_indexes]

    ai.classes_frequency(y_train)

    return x_train, y_train

def create_class_weight(labels_dict,mu=1):
    total = np.sum(list(labels_dict.values()))
    keys = labels_dict.keys()
    class_weight = dict()

    for key in keys:
        val = float(labels_dict.get(key))
        score = math.log(mu*total/val)
        class_weight[key] = score if score > 1.0 else 1.0

    return class_weight

def createX_Y(data):
    X, Y = [], []
    print(len(data[0]))
    with tqdm(total=len(data[0])) as pbar:
        for i in range(0, len(data[0])-1, 1):
            pbar.update(1)
            try:
                o = np.array(data[0][i]).astype(float)
                h = np.array(data[1][i]).astype(float)
                l = np.array(data[2][i]).astype(float)
                c = np.array(data[3][i]).astype(float)
                v = np.array(data[4][i]).astype(float)

                x_i = np.column_stack((o, h, l, c, v))
                y_i = data[3][i+1]
            except Exception as e:
                print(e)
                break

            if np.isnan(x_i).any() == False:
                X.append(x_i)
                Y.append(y_i)
    return X, Y


def shuffle_in_unison(a, b):
    # courtsey http://stackoverflow.com/users/190280/josh-bleecher-snyder
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b
