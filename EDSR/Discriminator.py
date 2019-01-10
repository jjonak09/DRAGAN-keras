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
from keras.initializers import RandomNormal,glorot_uniform
w_init = glorot_uniform()

def ResBlock(layer_input, out_channel):
    d = Conv2D(out_channel, 3, strides=1, padding='same',kernel_initializer=w_init)(layer_input)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(out_channel, 3, strides=1, padding='same',kernel_initializer=w_init)(d)
    d = Add()([d, layer_input])
    d = LeakyReLU(alpha=0.2)(d)
    return d


def Discriminator(input_shape, base=16):

    input = Input(shape=input_shape)
    h = Conv2D(base, 4, strides=2, padding="same",kernel_initializer=w_init)(input)
    h = LeakyReLU(alpha=0.2)(h)
    h = ResBlock(h, base)
    h = ResBlock(h, base)
    h = Conv2D(base*2, 4, strides=2, padding="same",kernel_initializer=w_init)(h)
    h = LeakyReLU(alpha=0.2)(h)
    h = ResBlock(h, base*2)
    h = ResBlock(h, base * 2)
    h = Conv2D(base*4, 4, strides=2, padding="same",kernel_initializer=w_init)(h)
    h = LeakyReLU(alpha=0.2)(h)
    h = ResBlock(h, base*4)
    h = ResBlock(h, base * 4)
    h = Conv2D(base*8, 3, strides=2, padding="same",kernel_initializer=w_init)(h)
    h = LeakyReLU(alpha=0.2)(h)
    h = ResBlock(h, base*8)
    h = ResBlock(h, base * 8)
    h = Conv2D(base*16, 3, strides=2, padding="same",kernel_initializer=w_init)(h)
    h = ResBlock(h, base*16)
    h = ResBlock(h, base * 16)
    h = Conv2D(base*32, 3, strides=2, padding="same",kernel_initializer=w_init)(h)
    h = LeakyReLU(alpha=0.2)(h)
    h = Flatten()(h)
    output = Dense(1)(h)

    return Model(inputs=input, outputs=output)


# def Dis_ResBlock(layer_input, out_channel,kernel_size=3, strides=1):
#     d = Conv2D(out_channel, kernel_size=kernel_size, strides=strides, padding='same',kernel_initializer=w_init)(layer_input)
#     d = LeakyReLU(alpha=0.2)(d)
#     return d

# def  Discriminator(input_shape,base=64):

#     input = Input(shape=input_shape)
#     h = input
#     h = Dis_ResBlock(h,base)
#     h = Dis_ResBlock(h,base,strides=2)
#     h = Dis_ResBlock(h,base*2)
#     h = Dis_ResBlock(h,base*2,strides=2)
#     h = Dis_ResBlock(h,base*4)
#     h = Dis_ResBlock(h,base*4,strides=2)
#     h = Dis_ResBlock(h,base*8)
#     h = Dis_ResBlock(h,base*8,strides=2)
#     h = Dense(base*16)(h)
#     h = LeakyReLU(alpha=0.2)(h)
#     h = Flatten()(h)
#     output = Dense(1)(h)

#     return Model(inputs=input,outputs=output)


if __name__ == '__main__':
    input_shape = (128, 128, 3)
    dis = Discriminator(input_shape)
    dis.summary()
