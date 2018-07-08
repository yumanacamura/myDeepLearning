import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation

test_nn = Sequential()
test_nn.add(Dense(4,input_dim=1))
test_nn.add(Activation('sigmoid'))
test_nn.add(Dense(2))
test_nn.add(Activation('softmax'))
test_nn.summary()
test_nn.compile(optimizer='adam',loss='categorical_crossentropy')
x = np.array([i for i in range(1,151)])
y = np.array([[0,1] if i%2 else [1,0] for i in range(1,151)])
test_nn.fit(x,y,epochs=500)
print(test_nn.predict(x))
