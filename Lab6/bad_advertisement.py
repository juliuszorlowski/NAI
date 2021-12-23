import sys
import cv2
import play_adv as adv

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


def is_cascade(cascade, frame, no1, no2):
    result = cascade.detectMultiScale(frame, no1, no2)
    if len(result) == 0:
        return False
    else:
        return True

def rects(cascade, frame, gray, no1, no2, no3, blue, green, red):
    result = cascade.detectMultiScale(gray, no1, no2)
    for (x,y,w,h) in result:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (red, green, blue), no3)

    return result


cap0 = video_capture(0)
# cap_ad = video_capture('./adv.mp4')

while True:

    _, frame_0 = cap0.read()
    # _, frame_ad = cap_ad.read()
    frame_gray = cv2.cvtColor(frame_0, cv2.COLOR_BGR2GRAY)

    # face_cascade = cascade('data/haarcascades/haarcascade_frontalface_default.xml')
    # face_rects = rects(face_cascade, frame_0, frame_gray, 1.3, 7, 2, 0, 255, 0)

    eye_cascade = cascade('data/haarcascades/haarcascade_eye.xml')

    adv.play_video(is_cascade(eye_cascade, frame_gray, 1.3, 20))

    eye_rects = rects(eye_cascade, frame_0, frame_gray, 1.3, 20, 2, 50, 255, 255)
    # print(is_cascade(eye_cascade, frame_gray, 1.3, 20))
    # print(eye_cascade.isAny())
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