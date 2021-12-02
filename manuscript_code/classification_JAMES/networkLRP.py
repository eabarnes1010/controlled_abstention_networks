"""Classification with abstention network architecture."""

import numpy as np

import keras.backend as K
from keras.layers import Dense, Activation
from keras import regularizers
from keras import metrics
from keras import optimizers
from keras.utils import to_categorical
from keras.models import Sequential
from keras.callbacks import LearningRateScheduler
from keras.models import load_model
import tensorflow.keras as keras
import tensorflow as tf

__author__ = "Elizabeth A. Barnes and Randal J. Barnes"
__date__ = "January 11, 2021"

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def defineNN(hidden, input_shape, output_shape, ridge_penalty=0., act_fun='relu', network_seed=99):
#     keras.backend.clear_session()  
    model = Sequential()

    # initialize first layer
    if hidden[0] == 0:
        # model is linear
        model.add(
            Dense(
                1,
                input_shape=(input_shape,),
                activation='linear',
                use_bias=True,
                kernel_regularizer=regularizers.l1_l2(l1=0.00, l2=ridge_penalty),
                bias_initializer=keras.initializers.RandomNormal(seed=network_seed),
                kernel_initializer=keras.initializers.RandomNormal(seed=network_seed)
            )
        )

    else:
        # model is a single node with activation function
        model.add(
            Dense(
                hidden[0],
                input_shape=(input_shape,),
                activation=act_fun,
                use_bias=True,
                kernel_regularizer=regularizers.l1_l2(l1=0.00, l2=ridge_penalty),
                bias_initializer=keras.initializers.RandomNormal(seed=network_seed),
                kernel_initializer=keras.initializers.RandomNormal(seed=network_seed)
            )
        )

        # initialize other layers
        for layer in hidden[1:]:
            model.add(
                Dense(
                    layer,
                    activation=act_fun,
                    use_bias=True,
                    kernel_regularizer=regularizers.l1_l2(l1=0.00, l2=0.00),
                    bias_initializer=keras.initializers.RandomNormal(seed=network_seed),
                    kernel_initializer=keras.initializers.RandomNormal(seed=network_seed)
                )
            )

    # initialize output layer
    model.add(
        Dense(
            output_shape,
            activation=None,
            use_bias=True,
            kernel_regularizer=regularizers.l1_l2(l1=0.00, l2=0.0),
            bias_initializer=keras.initializers.Zeros(),
            kernel_initializer=keras.initializers.Zeros()            
#             bias_initializer=keras.initializers.RandomNormal(seed=network_seed),
#             kernel_initializer=keras.initializers.RandomNormal(seed=network_seed)
        )
    )

    model.add(Activation('softmax'))
    return model

