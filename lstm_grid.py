import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import LSTM
from keras.optimizers import Adam, SGD, rmsprop, Nadam
from keras.initializers import *
from keras.regularizers import *
from keras.activations import softmax


def get_lstm_grid(input_shape=(1, 5, 60), num_classes=2, lr=0., dropout=0.):

    model = Sequential()
    #model.add(LSTM(128, batch_input_shape=input_shape, stateful=False, unroll=True, activation='sigmoid', return_sequences=True, dropout=dropout))
    model.add(LSTM(10, batch_input_shape=input_shape, stateful=False, unroll=True, activation='sigmoid', dropout=dropout, kernel_initializer='glorot_normal', kernel_regularizer=l2(0.01)))
    model.add(Dense(32, activation='sigmoid'))
    model.add(Dense(2, activation='sigmoid'))

    opt = Nadam()#SGD(lr=lr, decay=1e-5, momentum=0.9, nesterov=True)


    model.compile(loss='categorical_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'])

    return model
