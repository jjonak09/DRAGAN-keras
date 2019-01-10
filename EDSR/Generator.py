import keras.backend as K
import tensorflow as tf
from keras.layers import Input, Dense, Reshape, Flatten, Dropout,BatchNormalization
from keras.layers import Activation, Lambda, Add, Multiply
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import sigmoid, relu, softplus, tanh
from keras.layers.convolutional import Conv2D, Conv2DTranspose,UpSampling2D
from keras.models import Model, Sequential
from keras.initializers import RandomNormal
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
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


def ResBlock(channels, input_layer):
    h = Conv2D(channels, 3, strides=1, padding="same",kernel_initializer=conv_init)(input_layer)
    h = Activation('relu')(h)
    h = Conv2D(channels, 3, strides=1, padding="same",kernel_initializer=conv_init)(h)
    h = Lambda(lambda h: h * 0.1)(h)
    return Add()([h, input_layer])


def CBR(channels, input_layer):
    h = Conv2D(channels, 3, strides=1, padding="same")(input_layer)
    h = SubpixelConv2D(h)(h)
    h = BatchNormalization(momentum=0.8)(h)
    h = Activation('relu')(h)
    return h

# def CBR(channels, layer_input):
#     x = UpSampling2D(size=2)(layer_input)
#     x = Conv2D(channels,3,strides=1,padding="same",kernel_initializer=w_init)(x)
#     x = BatchNormalization(momentum=0.8)(x)
#     x = Activation('relu')(x)
#     return x


def Generator(z_dim, base=64):
    input = Input(shape=(z_dim,))
    x = Dense(base * 16 * 16)(input)
    x = Reshape((16, 16, base))(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Activation('relu')(x)
    r = ResBlock(base, x)
    r = ResBlock(base, r)
    r = ResBlock(base, r)
    r = ResBlock(base, r)
    r = ResBlock(base, r)
    r = ResBlock(base, r)
    h = Conv2D(base, 3, strides=1, padding="same",kernel_initializer=w_init)(r)
    h = Add()([h, x])
    h = CBR(base * 4, h)
    h = CBR(base * 4, h)
    h = CBR(base * 4, h)
    h = Conv2D(3, 9, strides=1, padding="same",kernel_initializer=w_init)(h)
    output = Activation('tanh')(h)
    model = Model(inputs=input, outputs=output)
    return model


if __name__ == '__main__':
    z_dim = 100
    gen = Generator(z_dim)
    gen.summary()
