from __future__ import print_function, division
import keras.backend as K
import tensorflow as tf
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import Activation,Lambda, Add
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import sigmoid,relu,softplus,tanh
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Model, Sequential
from keras.initializers import RandomNormal
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import argparse
# from Generator import Generator
from Discriminator import Discriminator
from RRDB import Generator
# -----------------
# parameters
# -----------------

parser = argparse.ArgumentParser(description="test")
parser.add_argument("--epoch", default=10000, type=int,
                    help="the number of epochs")
parser.add_argument("--save_interval", default=1, type=int,
                    help="the interval of snapshot")
parser.add_argument("--model_interval", default=10, type=int,
                    help="the interval of savemodel")
parser.add_argument("--batchsize", default=100, type=int, help="batch size")
parser.add_argument("--lam", default=5.0, type=float,
                    help="the weight of regularizer")

args = parser.parse_args()
epochs = args.epoch
save_interval = args.save_interval
model_interval = args.model_interval
batch_size = args.batchsize
_lambda = args.lam

z_dim = 128
img_shape = (128, 128, 3)
image_size = 128
channels = 3
lr_D = 2e-4
lr_G = 2e-4
b1 = 0.5
b2 = 0.99

gen = Generator(z_dim)
dis = Discriminator(img_shape)

# -----------------
# load dataset
# -----------------

X_train = np.load('./dataset128.npy')
X_train = np.float32(X_train)
X_train = X_train/127.5 - 1
X_train = np.expand_dims(X_train, axis=3)
X_train = X_train.reshape(
    X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[4])


# -------------------------
# compute grandient penalty
# -------------------------

dis_real = Input(shape=(image_size, image_size, channels))
noisev = Input(shape=(z_dim,))
dis_fake = gen(noisev)
valid = dis(dis_fake)

delta_input = K.placeholder(shape=(None, image_size, image_size, channels))
# alpha = K.random_uniform(
#     shape=[batch_size, 1, 1, 1],
#     minval=0.,
#     maxval=1.
# )

dis_mixed = Input(shape=(image_size, image_size, channels),
                  tensor=dis_real + delta_input)

loss_real = K.sum(K.softplus(-dis(dis_real))) / batch_size
loss_fake = K.sum(K.softplus(valid)) / batch_size

# dis_mixed_real = alpha * dis_real + ((1 - alpha) * dis_mixed)

grad_mixed = K.gradients(dis(dis_mixed), [dis_mixed])[0]
norm = K.sqrt(K.sum(K.square(grad_mixed), axis=[1, 2, 3]))
grad_penalty = K.mean(K.square(norm - 1))
loss_dis = loss_fake + loss_real + _lambda * grad_penalty

# ---------------------
# loss for discriminator
# ---------------------

training_updates = Adam(lr=lr_D, beta_1=b1,beta_2=b2).get_updates(
    dis.trainable_weights, [], loss_dis)
dis_train = K.function([dis_real, noisev, delta_input],
                       [loss_real, loss_fake],
                       training_updates)

# -----------------
# loss for generator
# -----------------

loss_gen = K.sum(K.softplus(-valid)) / batch_size

training_updates = Adam(lr=lr_G, beta_1=b1,beta_2=b2).get_updates(
    gen.trainable_weights, [], loss_gen)
gen_train = K.function([noisev],
                       [loss_gen],
                       training_updates)


# -----------------
# Training session
# -----------------

fixed_noise = np.random.normal(size=(16, z_dim))
batch = X_train.shape[0] // batch_size
gen_logit = np.ones((batch_size,image_size,image_size,channels))
for epoch in range(epochs):
    print("Epoch is", epoch)
    for index in range(batch):

        idx = np.random.randint(0, X_train.shape[0], batch_size)
        image_batch = X_train[idx]
        noise = np.random.normal(size=(batch_size, z_dim))
        delta = 0.5 * image_batch.std() * np.random.random(size=image_batch.shape)
        delta *= np.random.uniform(size=(batch_size, 1, 1, 1))
        errD_real, errD_fake = dis_train([image_batch, noise, delta])
        errD = errD_real - errD_fake

        errG, = gen_train([noise])
        print('%d/%d  Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
              % (index, batch, errD, errG, errD_real, errD_fake))

        if epoch % save_interval == 0 and index == 0:
            gen_imgs = gen.predict(fixed_noise)
            r, c = 4,4
            # Rescale images 0 - 1
            gen_imgs = 0.5 * gen_imgs + 0.5
            fig, axs = plt.subplots(r, c)
            cnt = 0
            for i in range(r):
                for j in range(c):
                    axs[i, j].imshow(gen_imgs[cnt])
                    axs[i, j].axis('off')
                    cnt += 1
            fig.savefig("result/%d.png" % epoch)
            plt.close()
            # if epoch % model_interval == 0 and index == 0:
            #     gen.save("DRAGAN_model/model-{}-epoch.h5".format(epoch))
