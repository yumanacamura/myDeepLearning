import gzip,pickle
import matplotlib.pyplot as plt
import numpy as np

import keras
from keras.models import Sequential, Model
from keras.layers import Input, Reshape, Dropout, BatchNormalization, ZeroPadding2D, Dense, Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers.merge import Concatenate
import tensorflow as tf
from keras.backend import tensorflow_backend
from keras.utils import np_utils

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
        self.l_dim = 10

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
        label_shape = (self.l_dim,)

        n_input = Input(noise_shape,name='noise_input')
        l_input = Input(label_shape,name='label_input')
        model = Concatenate()([n_input,l_input])
        model = Dense(128 * 28 * 28, activation="relu")(model)
        model = Reshape((28, 28, 128))(model)
        model = BatchNormalization(momentum=0.8)(model)
        model = Conv2D(128, kernel_size=3, padding="same")(model)
        model = Activation("relu")(model)
        model = BatchNormalization(momentum=0.8)(model)
                #model.add(UpSampling2D())
        model = Conv2D(64, kernel_size=3, padding="same")(model)
        model = Activation("relu")(model)
        model = BatchNormalization(momentum=0.8)(model)
        model = Conv2D(1, kernel_size=3, padding="same")(model)
        model = Activation("tanh",name='image')(model)

        gen = Model(inputs=[n_input,l_input], outputs=model)

        gen.summary()


        return gen

    def build_discriminator(self):
        img_shape = self.shape
        label_shape = (self.l_dim,)

        i_input = Input(img_shape,name='image_input')
        l_input = Input(label_shape,name='label_input')

        model = Conv2D(32, kernel_size=3, strides=2, padding="same")(i_input)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.25)(model)
        model = Conv2D(64, kernel_size=3, strides=2, padding="same")(model)
        model = ZeroPadding2D(padding=((0, 1), (0, 1)))(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.25)(model)
        model = BatchNormalization(momentum=0.8)(model)
        model = Conv2D(128, kernel_size=3, strides=2, padding="same")(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.25)(model)
        model = BatchNormalization(momentum=0.8)(model)
        model = Conv2D(256, kernel_size=3, strides=1, padding="same")(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.25)(model)

        model = Flatten()(model)
        model = Concatenate()([model,l_input])
        model = Dense(1, activation='sigmoid',name='valid')(model)

        dis = Model(inputs=[i_input,l_input], outputs=[model])

        dis.summary()

        return dis

    def build_combined(self):
        z = Input(shape=(self.z_dim,))
        l_g = Input(shape=(self.l_dim,))
        l_d = Input(shape=(self.l_dim,))
        img = self.generator([z,l_g])
        self.discriminator.trainable = False
        valid = self.discriminator([img,l_d])
        #model = Sequential([self.generator, self.discriminator])
        model = Model(inputs=[z,l_g,l_d],outputs=[valid])

        return model

    def train(self, iterations, batch_size=128, save_interval=50, model_interval=1000, check_noise=None, check_label=None, raw=5, col=5):

        X_train, labels = self.load_imgs()

        half_batch = int(batch_size / 2)

        X_train = (X_train.reshape(60000,28,28,1).astype(np.float32) - 127.5) / 127.5
        labels = np_utils.to_categorical(labels,self.l_dim)


        for iteration in range(iterations):

            # ------------------
            # Training Discriminator
            # -----------------
            idx = np.random.randint(0, X_train.shape[0], half_batch)

            imgs = X_train[idx]
            lbls = labels[idx]

            noise = np.random.uniform(-1, 1, (half_batch, self.z_dim))
            gen_lbls = np.random.randint(0,self.l_dim,half_batch)
            gen_lbls = np_utils.to_categorical(gen_lbls,self.l_dim)

            gen_imgs = self.generator.predict([noise,gen_lbls])

            d_loss_real = self.discriminator.train_on_batch([imgs,lbls], np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs,gen_lbls], np.zeros((half_batch, 1)))

            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # -----------------
            # Training Generator
            # -----------------

            noise = np.random.uniform(-1, 1, (batch_size, self.z_dim))
            gen_lbls = np.random.randint(0,self.l_dim,batch_size)
            gen_lbls = np_utils.to_categorical(gen_lbls,self.l_dim)



            g_loss = self.combined.train_on_batch([noise,gen_lbls,gen_lbls], np.ones((batch_size, 1)))

            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (iteration, d_loss[0], 100 * d_loss[1], g_loss))

            if iteration % save_interval == 0:
                self.save_imgs(iteration, check_noise, check_label, raw, col)
                #start = np.expand_dims(check_noise[0], axis=0)
                #end = np.expand_dims(check_noise[1], axis=0)
                #resultImage = #self.visualizeInterpolation(start=start, end=end)
                #cv2.imwrite("images/latent/" + "latent_{}.png".format(iteration), resultImage)
                if iteration % model_interval == 0:
                    self.generator.save("models/gan-{}-iter.h5".format(iteration))

    def save_imgs(self, iteration, check_noise, check_label, r, c):
        noise = check_noise
        gen_imgs = self.generator.predict([noise,check_label]).reshape(r*c,28,28)

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
        with gzip.open('../gan_mnist/data/mnist.pkl.gz', 'rb') as f:
            train, _ = pickle.load(f, encoding='bytes')

        return train

    def make_noise(self,num):
        return np.random.uniform(-1,1,(num,self.z_dim))

    def make_label(self,num):
        return np_utils.to_categorical(np.random.randint(0,self.l_dim,num),self.l_dim)
