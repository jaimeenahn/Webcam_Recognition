import cv2

resizedSize = (80, 60)

categories = ["badget", "orange", "human", "sweetpotato", "phone"]
numClasses = len(categories)


cap = cv2.VideoCapture(0)
cap.set(3, 640)  # 가로
cap.set(4, 480)  # 세로

width = int(cap.get(3))
height = int(cap.get(4))

fourcc = cv2.VideoWriter_fourcc(*'DIVX')

out = cv2.VideoWriter()

from keras.models import load_model
model = load_model("object/obj_5-model.h5")

while True:
    ret, frame = cap.read()

    k = cv2.waitKey(1)

    if k == 27:  # esc
        break
    else:
        imghat = frame.astype('float32') / 255.0
        resized = cv2.resize(frame, resizedSize)
        xhat = resized.reshape((1, 60, 80 ,3))
        print(xhat.shape)
        yhat = model.predict_classes(xhat)
        cv2.putText(frame, categories[yhat[0]], (100, 0), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
        cv2.imshow('WebCamVideo', frame)

cap.release()
cv2.destroyAllWindows()
