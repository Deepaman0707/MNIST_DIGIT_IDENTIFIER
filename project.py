from matplotlib.cbook import normalize_kwargs
import tensorflow as tf
mnist = tf.keras.datasets.mnist

#Destructing dataset and dividing it into train and test

(x_train, y_train), (x_test, y_test) = mnist.load_data()
import matplotlib.pyplot as plt
plt.imshow(x_train[0])
plt.show()
plt.imshow(x_train[0], cmap = plt.cm.binary)

#Before Normalizing

print(x_train[0])

#Normalizing the data

x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)
plt.imshow(x_train[0], cmap = plt.cm.binary)
plt.show()

#After Normalizing

print(x_train[0])

print(y_train[0])

#Resizing image to make it suitable for apply Convolution operation

import numpy as np
IMG_SIZE = 28
x_trainr = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
x_testr = np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
print ("Training Sample dimension", x_trainr.shape)
print ("Testing Sample dimension", x_testr.shape)

#Creating a Deep neural Network
#Training on 60000 samples of MNIST handwritten dataset

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D


#Creating a neural network

model = Sequential()

#First convolution layer

model.add(Conv2D(64, (3,3), input_shape = x_trainr.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))

#2nd convolution layer

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))

#3rd convolution layer

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))

#Fully connected layer

model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))

#Fully connected layer

model.add(Dense(32))
model.add(Activation("relu"))

#Last Fully connected layer

model.add(Dense(10))
model.add(Activation("softmax"))

model.summary()

print("Total Training Samples = ", len(x_trainr))

model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ['accuracy'])

#Training model

model.fit(x_trainr, y_train, epochs = 5, validation_split = 0.3)

#Evaluating on testiong dataset MNIST

test_loss, test_acc = model.evaluate(x_testr, y_test)
print("Test loss on 10000 test samples", test_loss)
print("Validation accuracy on 10000 test samples", test_acc)

predicions = model.predict([x_testr])

print(predicions)

print(np.argmax(predicions[0]))

plt.imshow(x_test[0])

print(np.argmax(predicions[128]))

plt.imshow(x_test[128])

import cv2

img = cv2.imread('eight.png')
plt.imshow(img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
resized = cv2.resize(gray, (28, 28), interpolation = cv2.INTER_AREA)
newing = tf.keras.utils.normalize(resized, axis = 1)
newing = np.array(newing).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

predicions = model.predict(newing)
print(np.argmax(predicions))

