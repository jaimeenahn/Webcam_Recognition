# Webcam_Recognition

## How to run?

1. saveVideo.py 실행하여 웹캡으로 학습시킬 피사체의 동영상을 찍는다.
   1. RECORD 시작 : s
   2. 해당 파일은 순서(*n*)대로 *output/video*에 *video(n).avi*형태로 저장된다.
   3. RECORD 종료 : e
2. labelFrame.py를 실행하여 녹화된 동영상을 라벨링한다.
   1. [battery:' b', orange: 'o', human: 'h', sweetpotato: 's', phone: 'p'] 중 하나의 카테고리를 선택하여 라벨링한다.
   2. 라벨링된 동영상을 압축된 사진(프레임)으로 *output/frame/해당카테고리* 에 저장된다.
3. makeDataset.py를 실행하여 동영상들을 모아 Dataset을 만든다.
   1. 압축된 사진을 cv2.imread를 이용하여  이미지를 읽어온다.
   2.  numpy의 형태로 이미지에 해당하는 자료 *X_train, X_test*와 라벨에 해당하는 자료 *Y_train, Y_test*로 저장된다.
   3. *object/*에 *obj_test.npy*의 형태로 저장한다.
4. learnDataset.py를 실행하여 CNN기법을 통해 학습을 진행한다.
   1. *X_train, Y_train*을 학습을 위한 *X_train, Y_train*과 유효성검사를 위한 *X_val, Y_val*로 나눠준다.
   2. Keras를 이용하여 CNN기법을 통해 학습이 진행된다.
   3. 학습이 완료된 모델은 *object/*에 *obj_5-model.h5*의 형태로 저장된다.
5. PredictLabel.py를 실행하여 웹캡에 보여지는 물체를 판별한다.(보조배터리, 귤, 사람, 고구마, 핸드폰 중)

## Download Link

*Obj_test.npy*의 예시의 경우 100MB를 초과하기 때문에 드라이브에서 다운부탁드립니다.

**LINK** : https://drive.google.com/file/d/12gjJZ6pkFEttOJtPt5n3WvMmeUon4wSU/view?usp=sharing