"""Regression with abstention network architecture."""

import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers

__author__ = "Elizabeth A. Barnes and Randal J. Barnes"
__date__ = "22 April 2021"

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def defineNN(hiddens, input_shape, output_shape, ridge_penalty=0., act_fun='relu', network_seed=99):
    """Define the Controlled Abstention Network.

        Arguments
        ---------

    """
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    # linear network only
    if hiddens[0] == 0:
        x = tf.keras.layers.Dense(
            1,
            activation='linear',
            use_bias=True,
            kernel_regularizer=regularizers.l1_l2(l1=0.00, l2=ridge_penalty),
            bias_initializer=tf.keras.initializers.RandomNormal(seed=network_seed),
            kernel_initializer=tf.keras.initializers.RandomNormal(seed=network_seed),
        )(x)
    else:
        # initialize first layer
        x = tf.keras.layers.Dense(
            hiddens[0],
            activation=act_fun,
            use_bias=True,
            kernel_regularizer=regularizers.l1_l2(l1=0.00, l2=ridge_penalty),
            bias_initializer=tf.keras.initializers.RandomNormal(seed=network_seed),
            kernel_initializer=tf.keras.initializers.RandomNormal(seed=network_seed),
        )(x)

        # initialize other layers
        for layer in hiddens[1:]:
            x = tf.keras.layers.Dense(
                layer,
                activation=act_fun,
                use_bias=True,
                kernel_regularizer=regularizers.l1_l2(l1=0.00, l2=0.0),
                bias_initializer=tf.keras.initializers.RandomNormal(seed=network_seed),
                kernel_initializer=tf.keras.initializers.RandomNormal(seed=network_seed),
            )(x)

    # set final output units separately
    mu_unit = tf.keras.layers.Dense(
        1,
        activation='linear',
        use_bias=True,
        bias_initializer=tf.keras.initializers.RandomNormal(seed=network_seed),
        kernel_initializer=tf.keras.initializers.RandomNormal(seed=network_seed),
        # bias_initializer=tf.keras.initializers.RandomUniform(
        #     minval=0.,maxval=1.,seed=network_seed),
        # kernel_initializer=tf.keras.initializers.RandomUniform(
        #     minval=0.,maxval=1.,seed=network_seed),
        )(x)

    sigma_unit = tf.keras.layers.Dense(
        1,
        activation='relu',       # constrain sigma to be positive
        use_bias=True,
        bias_initializer=tf.keras.initializers.RandomUniform(
            minval=.1, maxval=.5, seed=network_seed),
        kernel_initializer=tf.keras.initializers.RandomUniform(
            minval=.1, maxval=.5, seed=network_seed),
        # kernel_constraint = tf.keras.constraints.NonNeg(),
        # bias_constraint = tf.keras.constraints.NonNeg(),
        )(x)

    # final output layer
    output_layer = tf.keras.layers.concatenate([mu_unit, sigma_unit], axis=1)

    # finalize the model
    model = tf.keras.models.Model(inputs=inputs, outputs=output_layer)
    return model
