import cv2 as cv

eye_cascade = cv.CascadeClassifier(
    'data\haarcascades_cuda\haarcascade_eye.xml'
)

if eye_cascade.empty():
    raise IOError('Unable to load the cascade classifier xml file')

cap = cv.VideoCapture(0)

while True:
    _, frame = cap.read()

    eye_rects = eye_cascade.detectMultiScale(frame, 1.3, 5)

    for (x,y,w,h) in eye_rects:
        cv.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 3)

    cv.imshow('Eye Detector', frame)
    if cv.waitKey(1) == 27:
        break

cap.release()
cv.destroyAllWindows()