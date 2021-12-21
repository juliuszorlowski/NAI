import cv2


def cascade(path):
    result = cv2.CascadeClassifier(path)

    if result.empty():
        raise IOError('Unable to load the cascade classifier xml file')

    return result


def video_capture(number):
    result = cv2.VideoCapture(number)
    
    if not result.isOpened():
        print("Cannot open camera")
        exit()
    
    return result


def rects(cascade, frame, no1, no2, no3, blue, green, red):
    result = cascade.detectMultiScale(frame, no1, no2)

    for (x,y,w,h) in result:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (red, green, blue), no3)
    return result


cap0 = video_capture(0)

while True:

    _, frame0 = cap0.read()

    face_cascade = cascade('data/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cascade('data/haarcascades/haarcascade_eye.xml')

    face_rects = rects(face_cascade, frame0, 1.3, 5, 3, 0, 255, 0)
    eye_rects = rects(eye_cascade, frame0, 1.3, 5, 3, 0, 255, 0)

    cv2.imshow('Face and Eye Detector', frame0)
    if cv2.waitKey(1) == 27:
        break

cap0.release()
cv2.destroyAllWindows()