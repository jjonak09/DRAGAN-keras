import keras.backend as K
import tensorflow as tf
import numpy as np
from keras.layers import Input, Reshape, Flatten,Dense,BatchNormalization,PReLU
from keras.layers import Activation, Add,Lambda,AveragePooling2D,LeakyReLU,GlobalAvgPool2D
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model
from keras.initializers import RandomNormal,glorot_uniform
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



def Resblock_generator(layer_input,channels):
    # h1 = BatchNormalization(momentum=0.9)(layer_input)
    h1 = UpSampling2D(size=2)(layer_input)
    h1 = Conv2D(channels,3,strides=1,padding="same",kernel_initializer=conv_init)(h1)
    # h1 = SubpixelConv2D(h1)(h1)
    h1 = BatchNormalization(momentum=0.9)(h1)
    # h1 = Activation("relu")(h1)

    h2 = UpSampling2D(size=2)(layer_input)
    h2 = Conv2D(channels,1,strides=1,padding="valid",kernel_initializer=conv_init)(h2)
    # h2 = SubpixelConv2D(layer_input)(h2)
    # h2 = Activation("relu")(h2)

    return Add()([h2,h1])

def Generator(z_dim,base=64):
    input = Input(shape=(z_dim,))
    h = Dense(512*16*16)(input)
    h = Reshape((16,16,512))(h)
    # h = Resblock_generator(h,base*4)
    # h = Resblock_generator(h,base*4)
    h = Resblock_generator(h,base*4)
    h = Resblock_generator(h,base*2)
    h = Resblock_generator(h,base)
    h = BatchNormalization(momentum=0.9)(h)
    h = PReLU()(h)
    h = Conv2D(3,3,strides=1,padding='same')(h)
    output = Activation('tanh')(h)

    return Model(inputs=input,outputs=output)


def Resblock_discriminator(layer_input,channels):
    h1 = Conv2D(channels,3,strides=1,padding='same',kernel_initializer=conv_init)(layer_input)
    h1 = LeakyReLU(alpha=0.2)(h1)
    h1 = Conv2D(channels,3,strides=1,padding='same',kernel_initializer=conv_init)(h1)
    h1 = AveragePooling2D(pool_size=(2, 2))(h1)

    h2 = Conv2D(channels,1,strides=1,padding="valid",kernel_initializer=conv_init)(layer_input)
    h2 = LeakyReLU(alpha=0.2)(h2)
    h2 = AveragePooling2D(pool_size=(2, 2))(h2)

    return Add()([h2,h1])


def Discriminator(input_shape,base=64):
    input = Input(shape=input_shape)
    h = Resblock_discriminator(input,base)
    h = Resblock_discriminator(h,base*2)
    h = Resblock_discriminator(h,base*4)
    # h = Resblock_discriminator(h,base*4)
    h = Resblock_discriminator(h,base*8)
    h = LeakyReLU(alpha=0.2)(h)
    # h = GlobalAvgPool2D()(h)
    h = Flatten()(h)
    # h = Dense(1024)(h)
    output = Dense(1)(h)

    return Model(inputs=input,outputs=output)

# z_dim = 128
# gen = Generator(z_dim)
# gen.summary()
# input_shape=(128,128,3)
# dis = Discriminator(input_shape)
# dis.summary()
