import keras.backend as K
import tensorflow as tf
import numpy as np
from keras.layers import Input, Reshape, Flatten, Dropout, Concatenate,Dense
from keras.layers import Activation, Add,Lambda
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model
from keras.optimizers import Adam
import sys
import os
from keras.initializers import RandomNormal,glorot_uniform

conv_init = RandomNormal(0, 0.02)
w_init = glorot_uniform()

def SubpixelConv2D(input_shape, scale=2):

    # Copyright (c) 2017 Jerry Liu
    # Released under the MIT license
    # https://github.com/twairball/keras-subpixel-conv/blob/master/LICENSE

    def subpixel_shape(input_shape):
        dims = [input_shape[0],
                input_shape[1] * scale,
                input_shape[2] * scale,
                int(input_shape[3] / (scale ** 2))]
        output_shape = tuple(dims)
        return output_shape

    def subpixel(x):
        return tf.depth_to_space(x, scale)

    return Lambda(subpixel, output_shape=subpixel_shape)


def RDB(layer_input, base):
    h1 = Conv2D(base, 3, strides=1, padding="same",kernel_initializer=conv_init)(layer_input)
    h1 = Activation("relu")(h1)
    c1 = Add()([h1, layer_input])
    h2 = Conv2D(base, 3, strides=1, padding="same",kernel_initializer=conv_init)(c1)
    h2 = Activation("relu")(h2)
    c2 = Add()([h2, h1, layer_input])
    h3 = Conv2D(base, 3, strides=1, padding="same",kernel_initializer=conv_init)(c2)
    h3 = Activation("relu")(h3)
    c3 = Add()([h3, h2, h1, layer_input])
    x = Conv2D(base, 1, strides=1, padding="same",kernel_initializer=conv_init)(c3)
    return Add()([x, layer_input])


def CBR(channels, layer_input):
    x = UpSampling2D(size=2)(layer_input)
    x = Conv2D(channels,3,strides=1,padding="same",kernel_initializer=w_init)(x)
    x = Activation('relu')(x)
    return x


def Generator(z_dim, base=64):
    input = Input(shape=(z_dim,))
    h = Dense(base * 16 * 16)(input)
    h = Activation('relu')(h)
    h = Reshape((16,16, base))(h)
    s1 = Conv2D(base, 3, strides=1, padding="same",kernel_initializer=w_init)(h)
    s2 = Conv2D(base, 3, strides=1, padding="same",kernel_initializer=w_init)(s1)
    r1 = RDB(s2, base=base)
    r2 = RDB(r1, base=base)
    r3 = RDB(r2, base=base)
    c1 = Add()([r3, r2, r1])
    x = Conv2D(base, 1, strides=1, padding="same",kernel_initializer=w_init)(c1)
    x = Conv2D(base, 3, strides=1, padding="same",kernel_initializer=w_init)(x)
    c2 = Add()([x, s1])
    u = CBR(base *4,c2)
    u = CBR(base *4,u)
    u = CBR(base *4,u)
    output = Conv2D(3, 3, strides=1, padding="same",kernel_initializer=w_init)(u)
    output = Activation('tanh')(output)
    return Model(inputs=input, outputs=output)


if __name__ == '__main__':

    z_dim = 128
    gen = Generator(z_dim)
    gen.summary()
