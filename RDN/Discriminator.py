from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import Activation, Add
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.models import Model
from keras.initializers import RandomNormal
import numpy as np
from keras.initializers import RandomNormal

def ResBlock(layer_input, out_channel):
    d = Conv2D(out_channel, 3, strides=1, padding='same')(layer_input)
    d = LeakyReLU(alpha=0.2)(d)
    # d = Conv2D(out_channel, 3, strides=1, padding='same')(d)
    # d = Add()([d, layer_input])
    # d = LeakyReLU(alpha=0.2)(d)
    return d


def Discriminator(input_shape, base=16):

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
    h = LeakyReLU(alpha=0.2)(h)
    h = Flatten()(h)
    output = Dense(1)(h)

    return Model(inputs=input, outputs=output)


# def Dis_ResBlock(layer_input, out_channel,kernel_size=3, strides=1):
#     d = Conv2D(out_channel, kernel_size=kernel_size, strides=strides, padding='same',kernel_initializer=w_init)(layer_input)
#     d = LeakyReLU(alpha=0.2)(d)
#     return d
#
# def  Discriminator(input_shape,base=64):
#
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
#
#     return Model(inputs=input,outputs=output)


if __name__ == '__main__':
    input_shape = (128, 128, 3)
    dis = Discriminator(input_shape)
    dis.summary()
