from keras.models import Sequential
from keras.layers import Dropout, Activation, Dense
from keras.layers import Flatten, Convolution2D, MaxPooling2D
import numpy as np
import cv2

categories = ["battery", "orange", "human", "sweetpotato", "phone"]
numClasses = len(categories)

X_train, X_test, Y_train, Y_test = np.load('object/obj_test.npy')
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.0

num = len(X_train)

trainnum =(num // 1000)*1000

X_val = X_train[trainnum:]
Y_val = Y_train[trainnum:]
X_train = X_train[:trainnum]
Y_train = Y_train[:trainnum]


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

model.fit(X_train, Y_train, batch_size=32, nb_epoch=50,  validation_data=(X_val, Y_val) ,callbacks=[early_stopping])
score = model.evaluate(X_test, Y_test)
print('loss==>' ,score[0])
print('accuracy==>', score[1])

from keras.models import load_model
hdf5_file="object/obj_5-model.h5"
model.save(hdf5_file)

