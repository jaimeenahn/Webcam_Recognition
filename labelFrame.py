import cv2
import os

videoDir = 'output/video'
frameDir = 'output/frame'

resizedSize = (80, 60)


def labelFrame():
    global k
    path, dirs, files = next(os.walk(videoDir))

    for filePath in files:
        print(path + '/' + filePath)
        videoPath = path + '/' + filePath

        if not videoPath.endswith('.avi'):
            continue

        cap = cv2.VideoCapture(videoPath)
        cap.set(3, 640)  # 가로
        cap.set(4, 480)  # 세로

        isFirst = True
        category = 0

        while (cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break;

            cv2.imshow('Loaded Frame', frame)
            resized = cv2.resize(frame, resizedSize)
            k = 0

            if isFirst:
                k = cv2.waitKey(0)
                category = k
                isFirst = False
            else:
                k = cv2.waitKey(1)

            if k == 27:  # esc
                break
            elif category == ord('o'):
                path2, dirs2, files2 = next(os.walk(frameDir + '/orange'))
                file_count = len(files2)
                cv2.imwrite(frameDir + '/orange/' + str(file_count) + '.jpg', resized)
            elif category == ord('s'):
                path2, dirs2, files2 = next(os.walk(frameDir + '/sweetpotato'))
                file_count = len(files2)
                cv2.imwrite(frameDir + '/sweetpotato/' + str(file_count) + '.jpg', resized)
            elif category == ord('h'):
                path2, dirs2, files2 = next(os.walk(frameDir + '/human'))
                file_count = len(files2)
                cv2.imwrite(frameDir + '/human/' + str(file_count) + '.jpg', resized)
            elif category == ord('p'):
                path2, dirs2, files2 = next(os.walk(frameDir + '/phone'))
                file_count = len(files2)
                cv2.imwrite(frameDir + '/phone/' + str(file_count) + '.jpg', resized)
            elif category == ord('b'):
                path2, dirs2, files2 = next(os.walk(frameDir + '/battery'))
                file_count = len(files2)
                cv2.imwrite(frameDir + '/battery/' + str(file_count) + '.jpg', resized)
            elif category == ord('x'):
                continue

        cap.release()

    cv2.destroyAllWindows()


labelFrame()
