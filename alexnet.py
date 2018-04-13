import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.layers.advanced_activations import *
from keras.layers import GaussianNoise


# Paper: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
# Adapted from: https://github.com/jkh911208/cswithjames/blob/master/8_CIFAR10_alexnet.py

def get_alexNet(img_shape=(128, 64, 48, 3), num_classes=2):

    # AlexNet Define the Model
    model = Sequential()
    # model.add(Conv2D(96, (11,11), strides=(4,4), activation='relu', padding='same', input_shape=(img_height, img_width, channel,)))
    # for original Alexnet
    #model.add(Conv2D(96, (3,3), strides=(2,2), activation='LeakyReLU', padding='same', input_shape=img_shape))
    model.add(Conv2D(96, (3,3), strides=(2,2), padding='same', input_shape=img_shape))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(GaussianNoise(0.02))
    # Local Response normalization for Original Alexnet
    model.add(BatchNormalization())

    #model.add(Conv2D(256, (5,5), activation='LeakyReLU', padding='same'))
    model.add(Conv2D(256, (5,5), padding='same'))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
    # Local Response normalization for Original Alexnet
    model.add(BatchNormalization())

    #model.add(Conv2D(384, (3,3), activation='LeakyReLU', padding='same'))
    #model.add(Conv2D(384, (3,3), activation='LeakyReLU', padding='same'))
    #model.add(Conv2D(256, (3,3), activation='LeakyReLU', padding='same'))
    model.add(Conv2D(384, (3,3), padding='same'))
    model.add(LeakyReLU())
    model.add(Conv2D(384, (3,3), padding='same'))
    model.add(LeakyReLU())
    model.add(Conv2D(256, (3,3), padding='same'))
    model.add(LeakyReLU())

    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
    # Local Response normalization for Original Alexnet
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model
