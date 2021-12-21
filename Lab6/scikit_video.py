import skvideo.io
import skvideo.datasets
import numpy as np

# here you can set keys and values for parameters in ffmpeg
reader = skvideo.io.FFmpegReader(skvideo.datasets.bigbuckbunny())

skvideo.io.vwrite("outputvideo.mp4", reader)