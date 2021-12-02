"""Classification with abstention network architecture."""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential

__author__ = "Elizabeth A. Barnes and Randal J. Barnes"
__date__ = "January 11, 2021"

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def defineNN(hidden, input_shape, output_shape, ridge_penalty=0., act_fun='relu', network_seed=99):
#     tf.keras.backend.clear_session()  
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
                bias_initializer=tf.keras.initializers.RandomNormal(seed=network_seed),
                kernel_initializer=tf.keras.initializers.RandomNormal(seed=network_seed)
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
                bias_initializer=tf.keras.initializers.RandomNormal(seed=network_seed),
                kernel_initializer=tf.keras.initializers.RandomNormal(seed=network_seed)
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
                    bias_initializer=tf.keras.initializers.RandomNormal(seed=network_seed),
                    kernel_initializer=tf.keras.initializers.RandomNormal(seed=network_seed)
                )
            )

    # initialize output layer
    model.add(
        Dense(
            output_shape,
            activation=None,
            use_bias=True,
            kernel_regularizer=regularizers.l1_l2(l1=0.00, l2=0.0),
            bias_initializer=tf.keras.initializers.Zeros(),
            kernel_initializer=tf.keras.initializers.Zeros()            
#             bias_initializer=tf.keras.initializers.RandomNormal(seed=network_seed),
#             kernel_initializer=tf.keras.initializers.RandomNormal(seed=network_seed)
        )
    )

    model.add(tf.keras.layers.Softmax())
    return model

