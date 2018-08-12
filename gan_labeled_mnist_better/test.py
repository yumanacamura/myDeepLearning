import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from keras.utils import np_utils

FILENAME = './models/gan-200000-iter.h5'

model = load_model(FILENAME)

print('OK')
label = int(input())

while label:
    noise = np.random.uniform(-1,1,(1,100))
    label = np_utils.to_categorical(label,10).reshape((1,10))
    img = model.predict([noise,label]).reshape((28,28))
    img = 0.5 * img + 0.5
    print(img.shape)
    plt.imshow(img)
    plt.show()
    label = int(input())
