from picamera2.encoders import H264Encoder
from picamera2 import Picamera2, Preview
import time

# picam2 = Picamera2()
# video_config = picam2.create_video_configuration(main={"size": (640, 480)}, lores={"size": (640, 480)}, display="lores")
# picam2.configure(video_config)

# encoder = H264Encoder(bitrate=10000000)
# output = "test.mp4"

# picam2.start_recording(encoder, output)
# time.sleep(10)
# picam2.stop_recording()

picam2 = Picamera2()

picam2.start_and_record_video("test.mp4", duration=5)