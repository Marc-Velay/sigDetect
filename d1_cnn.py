import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Conv1D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, SGD, rmsprop, Nadam
from keras.initializers import *
from keras.layers.advanced_activations import *
from keras.layers import GaussianNoise



def get_1dcnn_grid(nb_features=60, nb_channels=4, num_classes=2, dropout_rate=0.6, lr=0.05):

    model = Sequential()
    model.add(Conv1D(nb_filter=8, filter_length=5, input_shape=(nb_features, nb_channels)))
    model.add(LeakyReLU(alpha=.0006))#Activation('LeakyReLU'))
    model.add(Flatten())
    model.add(Dropout(dropout_rate))
    model.add(Dense(16))#, activation='LeakyReLU'))
    model.add(LeakyReLU(alpha=.0006))
    #model.add(Dense(128, activation='sigmoid'))
    #model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(num_classes))
    #model.add(Activation('LeakyReLU'))
    model.add(LeakyReLU(alpha=.0006))


    optimizer = Nadam()#SGD(lr=lr, decay=1e-5, momentum=0.99, nesterov=True)

    model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

    return model, optimizer
