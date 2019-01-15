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
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class Incy:
    if K.backend() == 'tensorflow':
        K.set_image_data_format("channels_last")

    print(keras.__version__)

    def launching(self):
        epoch = 250
        batch_size = 10
        #learning_rate = 0.002
        input_dim = 3

        model = Sequential()
        model.add(Dense(64, input_dim=input_dim, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(1, activation='softmax'))

        model.compile(loss="binary_crossentropy",
                      optimizer="adamax", metrics=['accuracy'])

        # FIT AND EVALUATE ---- Manca ancora il dataset
        
        dataset = np.loadtxt("dataset.txt", delimiter=" ")
        X_train_CCC = dataset[:70]
        Y_train = [1] * 70
        X_test = dataset[70:]
        Y_test = [1]*30

        model.fit(X_train_CCC, Y_train, epochs=epoch, batch_size=batch_size)
        res = model.evaluate(X_test, Y_test)
        print("\n\nRESULT: %s: %.2f%%" % (model.metrics_names[1], res[1] * 100))
        
        model.save('accelerometer_inference.h5')

        output_names = [node.op.name for node in model.outputs]
        sess = tf.keras.backend.get_session()
        frozen_def = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, output_names)
        tflite_model = tf.contrib.lite.toco_convert(
            frozen_def, [inputs], output_names)

        with tf.gfile.GFile(tflite_graph, 'wb') as f:
            f.write(tflite_model)

if __name__ == '__main__':
    i=Incy()
    i.launching()
