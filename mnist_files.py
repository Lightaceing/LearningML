#Imports

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

def pre_process(x_train, y_train, x_test, y_test):
    #Pre processing the data
    n_pixels = x_train.shape[1] * x_train.shape[2]
    X_train_reshaped = x_train.reshape(x_train.shape[0], n_pixels).astype('float32')
    X_test_reshaped = x_test.reshape(x_test.shape[0], n_pixels).astype('float32')

    #normalizing 
    x_train_n = X_train_reshaped / 255
    x_test_n = X_test_reshaped/255

    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    n_classes = y_train.shape[1]

    return x_train_n, x_test_n, y_train, y_test, n_pixels, n_classes

def cnn_model(n_pixels):
    model = Sequential()
    model.add(Dense(n_pixels, input_dim=n_pixels, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train, X_test, Y_train, Y_test, n_pixels, n_classes = pre_process(X_train, Y_train, X_test, Y_test)

model = cnn_model(n_pixels)
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=15, batch_size=200, verbose=2)
score = model.evaluate(X_test, Y_test, verbose=2)