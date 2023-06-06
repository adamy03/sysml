"""
This file tests energy usage of Raspberry Picamera2 while it
takes a video at varying frame rates and resolutions.
"""
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput
import time

"""
Testing taking a video with varying resolutions, fps, and duration. 
"""
def test_camera_video(x_resolution, y_resolution, fps, num_seconds):
    # Initialize camera 
    print("Initializing camera...")
    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration())
    encoder = H264Encoder()
    output = FfmpegOutput("test_video.mp4")

    # Set resolution and frame rate
    picam2.video_configuration.size = (x_resolution, y_resolution)
    frame_duration = 1.0 / fps * 1000000
    picam2.video_configuration.controls.FrameDurationLimits = (frame_duration, frame_duration)
    
    # Take video
    print("Starting video...")
    picam2.start_recording(encoder, output)
    time.sleep(num_seconds)
    
    picam2.stop_recording()
    print("Stopped video...")

"""
Define execution of desired tests here:
"""
def run_tests():
    # Varying resolution with 25 fps and 10 sec
    test_camera_video(640, 480, 25, 10)
    test_camera_video(1280, 720, 25, 10)
    test_camera_video(1920, 1080, 25, 10)

if __name__ == '__main__':
    run_tests()
