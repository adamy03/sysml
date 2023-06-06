from picamera2 import Picamera2
from picamera2.encoders import H264Encoder, Quality
from picamera2.outputs import FfmpegOutput
import time

picam2 = Picamera2()
# picam2.configure(picam2.create_video_configuration())

# Choose video settings

# Resolution (size)
picam2.video_configuration.size = (640, 480)
# Frame rate: 25 fps
picam2.video_configuration.controls.FrameDurationLimits = (40000, 40000)

encoder = H264Encoder()
output = FfmpegOutput("test_video.mp4")
# quality = Quality.VERY_LOW

# Take video
picam2.start_recording(encoder, output, quality)
time.sleep(10)
picam2.stop_recording()
