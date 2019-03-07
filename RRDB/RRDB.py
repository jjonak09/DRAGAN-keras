import keras.backend as K
import tensorflow as tf
import numpy as np
from keras.layers import Input, Reshape, Flatten, Dropout, Concatenate,Dense,BatchNormalization
from keras.layers import Activation, Add,Lambda
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model
from keras.optimizers import Adam
import sys
import os
from keras.initializers import RandomNormal


conv_init = RandomNormal(0, 0.02)


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




def RRDB(layer_input,base,scale=0.1):
    h1 = RDB(layer_input,base)
    h1 = Lambda(lambda h: h *scale )(h1)
    h1 = Add()([h1,layer_input])
    h2 = RDB(h1,base)
    h2 = Lambda(lambda h: h *scale )(h2)
    h2 = Add()([h2,h1])
    h3 = RDB(h2,base)
    h3 = Lambda(lambda h: h *scale )(h3)
    h3 = Add()([h3,h2])
    h = Lambda(lambda h: h *scale )(h3)
    return Add()([h,layer_input])

# def CBR(channels, layer_input):
#     h = UpSampling2D(size=(2, 2))(layer_input)
#     h = Conv2D(channels, 3, strides=1, padding="same",kernel_initializer=conv_init)(h)
#     h = BatchNormalization(momentum=0.8)(h)
#     h = Activation('relu')(h)
#     return h


def CBR(channels, layer_input):
    h = Conv2D(channels, 3, strides=1, padding="same",kernel_initializer=conv_init)(layer_input)
    h = SubpixelConv2D(h)(h)
    h = BatchNormalization(momentum=0.8)(h)
    h = Activation('relu')(h)
    return h


def Generator(z_dim, base=64):
    input = Input(shape=(z_dim,))
    h = Dense(base * 16 * 16)(input)
    h = Reshape((16,16, base))(h)
    h = BatchNormalization(momentum=0.8)(h)
    h = Activation('relu')(h)
    r1 = RRDB(h, base=base)
    r2 = RRDB(r1, base=base)
    r3 = RRDB(r2, base=base)
    r4 = RRDB(r3, base=base)
    r5 = RRDB(r4, base=base)
    r5 = Add()([r5,r4,r3,r2,r1,h])
    r6 = RRDB(r5, base=base)
    r7 = RRDB(r6, base=base)
    r8 = RRDB(r7, base=base)
    h = Add()([r8,r7,r6,r5])
    h = Conv2D(base, 1, strides=1, padding="same",kernel_initializer=conv_init)(h)
    h = Conv2D(base, 3, strides=1, padding="same",kernel_initializer=conv_init)(h)
    h = BatchNormalization(momentum=0.8)(h)
    h = Activation('relu')(h)
    h = CBR(base *4,h)
    h = CBR(base *4,h)
    h = CBR(base *4,h)
    h = Conv2D(3, 3, strides=1, padding="same",kernel_initializer=conv_init)(h)
    output = Activation('tanh')(h)
    return Model(inputs=input, outputs=output)


if __name__ == '__main__':

    z_dim = 128
    gen = Generator(z_dim)
    gen.summary()
