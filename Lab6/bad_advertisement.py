import cv2
import skvideo.io
import mediapipe as mp



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
cap_ad = video_capture('./adv.mp4')

while True:

    _, frame_0 = cap0.read()
    _, frame_ad = cap_ad.read()

    face_cascade = cascade('data/haarcascades/haarcascade_frontalface_default.xml')
    face_rects = rects(face_cascade, frame_0, 1.3, 7, 2, 0, 255, 0)

    eye_cascade = cascade('data/haarcascades/haarcascade_eye.xml')
    eye_rects = rects(eye_cascade, frame_0, 1.3, 5, 3, 0, 255, 0)

    # glasses_cascade = cascade('data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
    # glasses_rects = rects(glasses_cascade, frame_0, 1.3, 5, 1, 0, 255, 255)

    # righteye_cascade = cascade('data/haarcascades/haarcascade_righteye_2splits.xml')
    # righteye_rects = rects(righteye_cascade, frame_0, 1.3, 7, 2, 255, 0, 0)

    # lefteye_cascade = cascade('data/haarcascades/haarcascade_lefteye_2splits.xml')
    # lefteye_rects = rects(lefteye_cascade, frame_0, 1.3, 7, 2, 0, 0, 255)

    cv2.imshow('Face and Eye Detector', frame_0)
    # cv2.imshow('Advertisement', frame_ad)
    if cv2.waitKey(1) == 27:
        break


cap0.release()
cv2.destroyAllWindows()