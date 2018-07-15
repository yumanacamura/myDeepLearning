import gzip,pickle
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D,  MaxPooling2D, Flatten
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

#make model
mnist = Sequential()
mnist.add(Convolution2D(10,(5,5),input_shape=(28,28,1),padding='same'))
mnist.add(MaxPooling2D(2,2))
mnist.add(Convolution2D(10,(5,5),padding='same'))
mnist.add(MaxPooling2D(2,2))
mnist.add(Convolution2D(10,(5,5),padding='same'))
mnist.add(MaxPooling2D(2,2))
mnist.add(Flatten())
mnist.add(Dense(10,activation='softmax'))
mnist.summary()
mnist.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#load data
with gzip.open('data/mnist.pkl.gz', 'rb') as f:
    (x_train, y_train), (x_test, y_test) = pickle.load(f, encoding='bytes')

#format data
x_train = x_train.reshape(60000,28,28,1)/255
x_test = x_test.reshape(10000,28,28,1)/255
y_train = np_utils.to_categorical(y_train,10)
y_test = np_utils.to_categorical(y_test,10)

#train
mnist.fit(x_train,y_train,epochs=1,batch_size=100,validation_data=(x_test,y_test),callbacks=[EarlyStopping()])

#save result
mnist.save('mnist.model')
