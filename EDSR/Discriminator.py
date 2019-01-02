import keras.backend as K
import tensorflow as tf
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import Activation, Lambda, Add, Multiply
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import sigmoid, relu, softplus, tanh
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Model, Sequential
from keras.initializers import RandomNormal
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt


def ResBlock(layer_input, out_channel):
    d = Conv2D(out_channel, 3, strides=1, padding='same')(layer_input)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(out_channel, 3, strides=1, padding='same')(d)
    d = Add()([d, layer_input])
    d = LeakyReLU(alpha=0.2)(d)
    return d


def Discriminator(input_shape, base=32):

    input = Input(shape=input_shape)
    h = Conv2D(base, 4, strides=2, padding="same")(input)
    h = LeakyReLU(alpha=0.2)(h)
    h = ResBlock(h, base)
    h = ResBlock(h, base)
    h = Conv2D(base*2, 4, strides=2, padding="same")(h)
    h = LeakyReLU(alpha=0.2)(h)
    h = ResBlock(h, base*2)
    h = ResBlock(h, base * 2)
    h = Conv2D(base*4, 4, strides=2, padding="same")(h)
    h = LeakyReLU(alpha=0.2)(h)
    h = ResBlock(h, base*4)
    h = ResBlock(h, base * 4)
    h = Conv2D(base*8, 3, strides=2, padding="same")(h)
    h = LeakyReLU(alpha=0.2)(h)
    h = ResBlock(h, base*8)
    h = ResBlock(h, base * 8)
    h = Conv2D(base*16, 3, strides=2, padding="same")(h)
    h = ResBlock(h, base*16)
    h = ResBlock(h, base * 16)
    h = Conv2D(base*32, 3, strides=2, padding="same")(h)
    h = Dense(base*32)(h)
    h = LeakyReLU(alpha=0.2)(h)
    h = Flatten()(h)
    output = Dense(1)(h)

    return Model(inputs=input, outputs=output)


input_shape = (128, 128, 3)
dis = Discriminator(input_shape)
dis.summary()
