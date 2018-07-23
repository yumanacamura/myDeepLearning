import gzip,pickle
import matplotlib.pyplot as plt
import numpy as np

import keras
from keras.models import Sequential, Model
from keras.layers import Input, Reshape, Dropout, BatchNormalization, ZeroPadding2D, Dense, Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
import tensorflow as tf
from keras.backend import tensorflow_backend

np.random.seed(0)
np.random.RandomState(0)
tf.set_random_seed(0)

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

class GAN:
    def __init__(self):
        self.shape = (28, 28, 1)
        self.z_dim = 100

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.generator = self.build_generator()
        # self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        #z = Input(shape=(self.z_dim,))
        #img = self.generator(z)

        #self.discriminator.trainable = False

        #valid = self.discriminator(img)

        #self.combined = Model(z, valid)
        self.combined = self.build_combined()
        self.combined.compile(loss='binary_crossentropy', optimizer='adam')

    def build_generator(self):
        noise_shape = (self.z_dim,)

        model = Sequential()

        model.add(Dense(128 * 28 * 28, activation="relu", input_shape=noise_shape))
        model.add(Reshape((28, 28, 128)))
        model.add(BatchNormalization(momentum=0.8))
        #model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        #model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(1, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):
        img_shape = self.shape

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=img_shape)
        validity = model(img)

        return Model(img, validity)

    def build_combined(self):
        z = Input(shape=(self.z_dim,))
        img = self.generator(z)
        self.discriminator.trainable = False
        valid = self.discriminator(img)
        #model = Sequential([self.generator, self.discriminator])
        model = Model(z,valid)

        return model

    def train(self, iterations, batch_size=128, save_interval=50, model_interval=1000, check_noise=None, raw=5, col=5):

        X_train, labels = self.load_imgs()

        half_batch = int(batch_size / 2)

        X_train = (X_train.reshape(60000,28,28,1).astype(np.float32) - 127.5) / 127.5

        for iteration in range(iterations):

            # ------------------
            # Training Discriminator
            # -----------------
            idx = np.random.randint(0, X_train.shape[0], half_batch)

            imgs = X_train[idx]

            noise = np.random.uniform(-1, 1, (half_batch, self.z_dim))

            gen_imgs = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))

            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # -----------------
            # Training Generator
            # -----------------

            noise = np.random.uniform(-1, 1, (batch_size, self.z_dim))

            g_loss = self.combined.train_on_batch(noise, np.ones((batch_size, 1)))

            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (iteration, d_loss[0], 100 * d_loss[1], g_loss))

            if iteration % save_interval == 0:
                self.save_imgs(iteration, check_noise, raw, col)
                #start = np.expand_dims(check_noise[0], axis=0)
                #end = np.expand_dims(check_noise[1], axis=0)
                #resultImage = #self.visualizeInterpolation(start=start, end=end)
                #cv2.imwrite("images/latent/" + "latent_{}.png".format(iteration), resultImage)
                if iteration % model_interval == 0:
                    self.generator.save("models/gan-{}-iter.h5".format(iteration))

    def save_imgs(self, iteration, check_noise, r, c):
        noise = check_noise
        gen_imgs = self.generator.predict(noise).reshape(r*c,28,28)

        # 0-1 rescale
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig('images/mnist_%d.png' % iteration)

        plt.close()

    def load_imgs(self):
        #load data
        with gzip.open('data/mnist.pkl.gz', 'rb') as f:
            train, _ = pickle.load(f, encoding='bytes')

        return train

    def make_noise(self,num):
        return np.random.uniform(-1,1,(num,self.z_dim))
