# import tensorflow as tf
import keras.backend as K
import numpy as np
from keras.layers import Input, Reshape, Flatten, Dropout, concatenate,Dense,BatchNormalization
from keras.layers import Activation, Add,Lambda,Concatenate
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model
from keras.optimizers import Adam
import sys
import os
from keras.initializers import RandomNormal

conv_init = RandomNormal(0, 0.02)

# def SubpixelConv2D(input_shape, scale=2):
#
#     # Copyright (c) 2017 Jerry Liu
#     # Released under the MIT license
#     # https://github.com/twairball/keras-subpixel-conv/blob/master/LICENSE
#
#     def subpixel_shape(input_shape):
#         dims = [input_shape[0],
#                 input_shape[1] * scale,
#                 input_shape[2] * scale,
#                 int(input_shape[3] / (scale ** 2))]
#         output_shape = tuple(dims)
#         return output_shape
#
#     def subpixel(x):
#         return tf.depth_to_space(x, scale)
#
#     return Lambda(subpixel, output_shape=subpixel_shape)


def RDB(layer_input, base):
    h1 = Conv2D(base, 3, strides=1, padding="same",kernel_initializer=conv_init)(layer_input)
    h1 = Activation('relu')(h1)
    c1 = concatenate([h1,layer_input],axis=3)
    h2 = Conv2D(base, 3, strides=1, padding="same",kernel_initializer=conv_init)(c1)
    h2 = Activation('relu')(h2)
    c2 = concatenate([h2,h1,layer_input],axis=3)
    h3 = Conv2D(base, 3, strides=1, padding="same",kernel_initializer=conv_init)(c2)
    h3 = Activation('relu')(h3)
    c3 = concatenate([h3,h2,h1,layer_input],axis=3)
    # h4 = Conv2D(base, 3, strides=1, padding="same",kernel_initializer=conv_init)(c3)
    # h4 = Activation('relu')(h4)
    # c4 = concatenate([h4,h3,h2,h1,layer_input],axis=3)
    x = Conv2D(base, 1, strides=1, padding="same",kernel_initializer=conv_init)(c3)
    return Add()([x, layer_input])


def CBR(channels, layer_input):
    x = UpSampling2D(size=2)(layer_input)
    x = Conv2D(channels,3,strides=1,padding="same",kernel_initializer=conv_init)(x)
    # x = BatchNormalization(momentum=0.9)(x)
    # x = Activation('relu')(x)
    return x

# def CBR(channels, layer_input):
#     h = Conv2D(channels, 3, strides=1, padding="same",kernel_initializer=conv_init)(layer_input)
#     h = SubpixelConv2D(h)(h)
#     h = BatchNormalization(momentum=0.8)(h)
#     h = Activation('relu')(h)
#     return h




def Generator(z_dim, base=64):
    input = Input(shape=(z_dim,))
    h = Dense(base * 16 * 16)(input)
    h = Reshape((16,16, base))(h)
    h = BatchNormalization(momentum=0.9)(h)
    h = Activation('relu')(h)
    s1 = Conv2D(base, 3, strides=1, padding="same",kernel_initializer=conv_init)(h)
    s2 = Conv2D(base, 3, strides=1, padding="same",kernel_initializer=conv_init)(s1)
    r1 = RDB(s2, base=base)
    r2 = RDB(r1, base=base)
    r3 = RDB(r2, base=base)
    concate = concatenate([r2,r1],axis=3)
    h = Conv2D(base, 1, strides=1, padding="same",kernel_initializer=conv_init)(concate)
    h = Conv2D(base, 3, strides=1, padding="same",kernel_initializer=conv_init)(h)
    c2 = Add()([h, s1])
    u = CBR(base *4,c2)
    u = CBR(base *4,u)
    u = CBR(base *4,u)
    output = Conv2D(3, 3, strides=1, padding="same",kernel_initializer=conv_init)(u)
    output = Activation('tanh')(output)
    return Model(inputs=input, outputs=output)


if __name__ == '__main__':

    z_dim = 128
    gen = Generator(z_dim)
    gen.summary()
