import sys
import cv2

from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QIcon, QPalette
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout,
                             QPushButton, QSlider, QStyle, QVBoxLayout,
                             QWidget)
from random import randrange

###################### Recognition #############################

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

###################### Player #############################
class Window(QWidget):
    def __init__(self, filename):
        super().__init__()

        self.setWindowIcon(QIcon('v_player1.png'))
        self.setWindowTitle('Player')
        self.setGeometry(350, 100, 700, 500)

        p = self.palette()
        p.setColor(QPalette.Window, Qt.black)
        self.setPalette(p)

        self.create_player(filename)


    def create_player(self, filename):
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)

        videowidget = QVideoWidget()


        self.open_file(filename)

        hbox = QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)

        vbox = QVBoxLayout()

        vbox.addWidget(videowidget)

        vbox.addLayout(hbox)

        self.mediaPlayer.setVideoOutput(videowidget)

        self.setLayout(vbox)


    def open_file(self, filename):
        self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(filename)))


    def play_video(self, bool):
        if bool:
            self.mediaPlayer.play()
        else:
            self.mediaPlayer.pause()


cap0 = video_capture(0)

app = QApplication(sys.argv)
window = Window('./adv.mp4')
window.show()
while True:

    _, frame_0 = cap0.read()
    frame_gray = cv2.cvtColor(frame_0, cv2.COLOR_BGR2GRAY)

    eye_cascade = cascade('data/haarcascades/haarcascade_eye.xml')
    window.play_video(is_cascade(eye_cascade, frame_gray, 1.3, 20))

    eye_rects = rects(eye_cascade, frame_0, frame_gray, 1.3, 20, 2, 50, 255, 255)

    if cv2.waitKey(1) == 27:
        break


cap0.release()
cv2.destroyAllWindows()
sys.exit(app.exec_())