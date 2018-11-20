#from _future_ import print_function

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation,Conv2D, MaxPooling2D, Input,Flatten,Dropout
from keras import optimizers
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

input_width = 28
input_height = 28
input_channels = 1
input_shape = (input_height,input_width,1)
num_classes = 10

x_train.shape
x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_train.shape
x_test = x_test.reshape(x_test.shape[0],28,28,1)
x_test.shape

model1 = Sequential()

model1.add(Conv2D(32, kernel_size=(5, 5),activation='relu',padding='SAME',input_shape=input_shape,use_bias=True))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Conv2D(64, (5, 5), activation='relu',padding='SAME',use_bias=True))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Flatten())
model1.add(Dense(num_classes,activation="softmax"))

adam = optimizers.Adam(lr=0.01)
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model1.summary()

model1.fit(x_train, keras.utils.to_categorical(y_train, 10),batch_size=600,epochs=30)

score = model1.evaluate(x_test,keras.utils.to_categorical(y_test, 10) ,batch_size=600)
score
print('test accuracy:', score[1])

#save the model

model_json = model1.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model1.save_weights("model.h5")