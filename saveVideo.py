import cv2
import os

videoDir = 'output/video'
frameDir = 'output/frame'


def saveVideo():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # 가로
    cap.set(4, 480)  # 세로

    width = int(cap.get(3))
    height = int(cap.get(4))
    print(width)
    print(height)

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    path, dirs, files = next(os.walk(videoDir))
    videoNum = len(files)
    isSaving = False

    out = cv2.VideoWriter()
    print(out)
    while True:
        ret, frame = cap.read()
        cv2.imshow('WebCamVideo', frame)

        k = cv2.waitKey(1)

        if k == 27:  # esc
            break
        elif k == ord('s'):#press start to record
            if not isSaving:
                isSaving = True
                print("start RECORDING")
                out = cv2.VideoWriter(videoDir + '/video' + str(videoNum) + '.avi', fourcc, 20, (width, height))
        elif k == ord('e'):#press end to record
            if isSaving:
                isSaving = False
                out.release()
                videoNum += 1
                print("END")

        if isSaving:
            out.write(frame)

    cap.release()
    cv2.destroyAllWindows()


saveVideo()
