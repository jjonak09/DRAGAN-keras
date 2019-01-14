import keras.backend as K
import numpy as np
from keras.layers import Input, Reshape, Flatten, Dropout, Concatenate,Dense,BatchNormalization
from keras.layers import Activation, Add,Lambda
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model
from keras.optimizers import Adam
import sys
import os
from keras.initializers import RandomNormal,glorot_uniform

conv_init = RandomNormal(0, 0.02)
w_init = glorot_uniform()



def RDB(layer_input, base):
    h1 = Conv2D(base, 3, strides=1, padding="same",kernel_initializer=conv_init)(layer_input)
    h1 = BatchNormalization(momentum=0.8)(h1)
    h1 = Activation("relu")(h1)
    c1 = Add()([h1, layer_input])
    h2 = Conv2D(base, 3, strides=1, padding="same",kernel_initializer=conv_init)(c1)
    h2 = BatchNormalization(momentum=0.8)(h2)
    h2 = Activation("relu")(h2)
    c2 = Add()([h2, h1, layer_input])
    h3 = Conv2D(base, 3, strides=1, padding="same",kernel_initializer=conv_init)(c2)
    h3 = BatchNormalization(momentum=0.8)(h3)
    h3 = Activation("relu")(h3)
    c3 = Add()([h3, h2, h1, layer_input])
    x = Conv2D(base, 1, strides=1, padding="same",kernel_initializer=conv_init)(c3)
    return Add()([x, layer_input])


def CBR(channels, layer_input):
    x = UpSampling2D(size=2)(layer_input)
    x = Conv2D(channels,3,strides=1,padding="same",kernel_initializer=w_init)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Activation('relu')(x)
    return x




def Generator(z_dim, base=64):
    input = Input(shape=(z_dim,))
    h = Dense(base * 16 * 16)(input)
    h = Reshape((16,16, base))(h)
    h = BatchNormalization(momentum=0.8)(h)
    h = Activation('relu')(h)
    s1 = Conv2D(base, 3, strides=1, padding="same",kernel_initializer=w_init)(h)
    s2 = Conv2D(base, 3, strides=1, padding="same",kernel_initializer=w_init)(s1)
    r1 = RDB(s1, base=base)
    r2 = RDB(r1, base=base)
    r3 = RDB(r2, base=base)
    c1 = Add()([r3, r2, r1])
    h = Conv2D(base, 1, strides=1, padding="same",kernel_initializer=w_init)(c1)
    h = Conv2D(base, 3, strides=1, padding="same",kernel_initializer=w_init)(h)
    h = BatchNormalization(momentum=0.8)(h)
    h = Activation('relu')(h)
    c2 = Add()([h, s1])
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
