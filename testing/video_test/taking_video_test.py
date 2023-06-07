"""
This file tests energy usage of Raspberry Picamera2 while it
takes a video at varying frame rates and resolutions.
"""
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput
import time

"""
Initializes the camera
"""
def init_camera():
    # Initialize camera 
    print("Initializing camera...")
    return Picamera2()  

"""
Testing taking a video with varying resolutions, fps, and duration. 
"""
def test_camera_video(picam2, x_resolution, y_resolution, fps, num_seconds, file_name):
    
    # Initialize video settings
    encoder = H264Encoder()
    output = FfmpegOutput(file_name)

    # Set resolution and frame rate
    picam2.video_configuration.size = (x_resolution, y_resolution)
    picam2.set_controls({"FrameRate": fps})

    # Take video
    print("Starting video...")
    picam2.start_recording(encoder, output)
    time.sleep(num_seconds)
    print("Resolution: " + str(picam2.video_configuration.size))
    print(picam2.video_configuration.controls)
          
    picam2.stop_recording()
    print("Stopped video...")
    
    # Wait after each test
    time.sleep(3)

"""
Define execution of desired tests here:
"""
def run_tests():
    # Initialize camera
    picam2 = init_camera()
    
    # Varying resolution with 30 fps and 10 sec
    test_camera_video(picam2, 64, 64, 30, 10, "64x64_vid.mp4")
    test_camera_video(picam2, 640, 480, 30, 10, "640x480_vid.mp4")
    test_camera_video(picam2, 1920, 1080, 30, 10, "1920x1080_vid.mp4")

if __name__ == '__main__':
    run_tests()
