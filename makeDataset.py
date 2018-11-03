import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split

frameDir = 'output/frame'
categories = ["battery", "orange", "human", "sweetpotato", "phone"]
numClasses = len(categories)

X = []
Y = []

for idx, category in enumerate(categories):
    label = [0 for i in range(numClasses)]
    label[idx] = 1

    imageDir = frameDir + '/' + category + '/'
    for path, dir, file in os.walk(imageDir):
        for filename in file:
            filePath = imageDir + filename
            print(filePath)

            if not filePath.endswith('.jpg'):
                continue

            img = cv2.imread(filePath)
            X.append(img) #X.append(img)
            Y.append(label)

X = np.array(X)
Y = np.array(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
xy = (X_train, X_test, Y_train, Y_test)

np.save("object/obj_test.npy", xy)
