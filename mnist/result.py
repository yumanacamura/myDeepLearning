import gzip,pickle
import matplotlib.pyplot as plt
import numpy as np
#import keras
from keras.models import load_model

#load trained model
model = load_model('mnist.model')

#load data
with gzip.open('data/mnist.pkl.gz', 'rb') as f:
    _, (x_test, _) = pickle.load(f, encoding='bytes')

#format data
x_te = x_test.reshape(10000,28,28,1)/255

#show result
for i in range(100):
    data_index = np.random.randint(10000)
    predicted = model.predict(x_te[data_index].reshape(1,28,28,1))
    print('{}'.format(np.argmax(predicted)))
    plt.imshow(x_test[data_index],cmap=plt.cm.gray_r)
    plt.show()
