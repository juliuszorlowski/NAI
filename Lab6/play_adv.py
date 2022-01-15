import sys

from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QIcon, QPalette
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout,
                             QPushButton, QSlider, QStyle, QVBoxLayout,
                             QWidget)
import bad_advertisement as Bad
from random import randrange

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
        

        # self.playBtn = QPushButton()
        # self.playBtn.setEnabled(True)
        # self.playBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        # self.playBtn.clicked.connect(self.play_video)

        
        hbox = QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)

        # hbox.addWidget(self.playBtn)

        vbox = QVBoxLayout()

        vbox.addWidget(videowidget)

        vbox.addLayout(hbox)

        self.mediaPlayer.setVideoOutput(videowidget)

        self.setLayout(vbox)

        self.play_video(randrange(1))


    def open_file(self, filename):
        self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(filename)))


    def play_video(self, bool):
        if bool:
            self.mediaPlayer.play()
        else:
            self.mediaPlayer.pause()

def __main__():
    app = QApplication(sys.argv)
    window = Window('./adv.mp4') 
    window.show()
    sys.exit(app.exec_()) 