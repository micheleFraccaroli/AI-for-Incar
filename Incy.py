import math
import os
import random
from time import time

import numpy as np

import keras
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.layers import (Activation, Dense, Dropout, Flatten)
from keras.models import Sequential
from keras.optimizers import (Adam, Adamax)
from labels_gen import labels_gen as lg

class Incy:
    if K.backend() == 'tensorflow':
        K.set_image_data_format("channels_last")

    def train_network(self):
        epoch = 500
        batch_size = 69
        learning_rate = math.exp(-6)
        input_dim = 3

        # NEURAL NET --------------------------------------------------------------------

        model = Sequential()
        model.add(Dense(96, input_dim=input_dim, activation='relu'))
        model.add(Dense(96, activation='relu'))
        model.add(Dense(96, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(96, activation='relu'))
        model.add(Dense(96, activation='relu'))
        model.add(Dense(96, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(2, activation='softmax'))

        opt = Adamax(lr = learning_rate)
        model.compile(loss="sparse_categorical_crossentropy",
                      optimizer=opt, metrics=['accuracy'])

        # FIT AND EVALUATE --------------------------------------------------------------
        
        lgen = lg()

        train_data = np.loadtxt("Training/train_data.txt", delimiter=" ")
        label_train = lgen.generate(
            "ACC_Yauto/LOG_ACCELEROMETRO.txt", "ACC_Nauto/LOG_ACCELEROMETRO.txt")
        
        test_data = np.loadtxt("Test/test_data.txt", delimiter=" ")
        label_test = lgen.generate(
            "ACC_Yauto_test/Test_Auto.txt", "ACC_Nauto_test/Test_AutoN.txt")

        model.fit(train_data, label_train, epochs=epoch, batch_size=batch_size)
        res = model.evaluate(test_data, label_test)
        print("\n\nRESULT: %s: %.2f%%" % (model.metrics_names[1], res[1] * 100))
        
        model.save('accelerometer_inference.h5')

if __name__ == '__main__':
    i=Incy()
    i.train_network()
