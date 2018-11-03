from keras.models import Sequential
from keras.layers import Dropout, Activation, Dense
from keras.layers import Flatten, Convolution2D, MaxPooling2D
from keras.utils import np_utils
import numpy as np
import cv2

categories = ["badget", "orange", "human", "sweetpotato", "phone"]
numClasses = len(categories)

X_train, X_test, Y_train, Y_test = np.load('object/obj.npy')
print('Xtrina_shape', Y_train.shape)
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0


X_val = X_train[2000:]
Y_val = Y_train[2000:]
X_train = X_train[:2000]
Y_train = Y_train[:2000]
print('Y_train.shape', Y_train)
"""
Y_train = np_utils.to_categorical(Y_train)
Y_val = np_utils.to_categorical(Y_val, 5)
Y_test = np_utils.to_categorical(Y_test, 5)"""

print('Xtrina_shape', X_train.shape)
print('X_train[0].shape', X_train[0].shape)
print('Y_train.shape', Y_train[0])

model = Sequential()
model.add(Convolution2D(32, (3, 3), padding='same', input_shape=X_train[0].shape,))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(numClasses))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping() # 조기종료 콜백함수 정의
#hist = model.fit(X_train, Y_train, epochs=3000, batch_size=10, validation_data=(X_val, Y_val), callbacks=[early_stopping])

model.fit(X_train, Y_train, batch_size=32, nb_epoch=50,  validation_data=(X_val, Y_val) ,callbacks=[early_stopping])
score = model.evaluate(X_test, Y_test)
print('loss==>' ,score[0])
print('accuracy==>', score[1])

from keras.models import load_model
hdf5_file="object/obj_5-model.h5"
model.save(hdf5_file)

"""
from keras.models import Sequential
from keras.layers import Dropout, Activation, Dense
from keras.layers import Flatten, Convolution2D, MaxPooling2D
import h5py, os
import numpy as np
import cv2

categories = ["book", "cup", "human", "monitor", "phone"]
numClasses = len(categories)

X_train, X_test, Y_train, Y_test = np.load('object/obj.npy')
print('Xtrain_shape', X_train.shape)

model = Sequential()
model.add(Convolution2D(32, (3, 3), padding='same', input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, (3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(numClasses))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
#model.fit(X_train, Y_train, batch_size=32, epochs=50)
model.fit(X_train, Y_train, batch_size=32, epochs=5)
score = model.evaluate(X_test, Y_test)
print('loss==>' ,score[0])
print('accuracy==>', score[1])

hdf5_file="object/obj_5-model.hdf5"
model.save_weights(hdf5_file)"""