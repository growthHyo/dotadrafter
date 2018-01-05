from keras.models import Sequential, Model
from keras.layers import Dense, Activation, merge, Input, concatenate
from keras.layers.core import Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam, SGD, Adagrad
import os.path
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.utils import multi_gpu_model
from keras import regularizers

WEIGHTS_FILE = 'data/weights.h5'
WEIGHT_DECAY = 0.0001

def get_model():
    inp = Input(shape=(400,))

    d1 = Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(WEIGHT_DECAY))
    out = d1(inp)
    # out = Dropout(0.1)(out)
    
    d2 = Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(WEIGHT_DECAY))
    out = d2(out)
    # out = Dropout(0.1)(out)
    
    d3 = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(WEIGHT_DECAY))
    out = d3(out)
    # out = Dropout(0.1)(out)
   
    final = Dense(2, activation='relu', kernel_regularizer=regularizers.l2(WEIGHT_DECAY))
    out = final(out)
    out = Activation("softmax")(out)

    model = Model(inputs=inp, outputs=out)
    if os.path.isfile(WEIGHTS_FILE):
        model.load_weights(WEIGHTS_FILE)

    # model.compile(optimizer=SGD(momentum=0.5, lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=Adagrad(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def save_model(model):
    model.save_weights(WEIGHTS_FILE)

def reload_model(model):
    if os.path.isfile(WEIGHTS_FILE):
        model.load_weights(WEIGHTS_FILE)
