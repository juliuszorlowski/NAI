import sys

from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QIcon, QPalette
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout,
                             QPushButton, QSlider, QStyle, QVBoxLayout,
                             QWidget)


class Window(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowIcon(QIcon('v_player1.png'))
        self.setWindowTitle('Player')
        self.setGeometry(350, 100, 700, 500)

        p = self.palette()
        p.setColor(QPalette.Window, Qt.black)
        self.setPalette(p)

        self.create_player()


    def create_player(self):
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)

        videowidget = QVideoWidget()


        self.open_file('./adv.mp4')
        

        self.playBtn = QPushButton()
        self.playBtn.setEnabled(True)
        self.playBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playBtn.clicked.connect(self.play_video)
        
        hbox = QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)

        hbox.addWidget(self.playBtn)

        vbox = QVBoxLayout()

        vbox.addWidget(videowidget)

        vbox.addLayout(hbox)

        self.mediaPlayer.setVideoOutput(videowidget)

        self.setLayout(vbox)


    def open_file(self, filename):
        self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(filename)))


    def play_video(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()

        else:
            self.mediaPlayer.play()


app = QApplication(sys.argv)
window = Window() 
window.show()
sys.exit(app.exec_()) 
