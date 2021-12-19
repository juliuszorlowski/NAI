import cv2

face_cascade = cv2.CascadeClassifier(
    'data\haarcascades\haarcascade_frontalface_default.xml'
)

if face_cascade.empty():
    raise IOError('Unable to load the cascade classifier xml file')

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()

    face_rects = face_cascade.detectMultiScale(frame, 1.3, 5)

    for (x,y,w,h) in face_rects:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 3)

    cv2.imshow('Face Detector', frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()