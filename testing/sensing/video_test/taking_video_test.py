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


def test_camera_video(x_resolution, y_resolution, num_seconds, file_name):
    # Initialize camera
    print("Initializing camera...")
    picam2 = Picamera2()

    # Initialize video properties
    encoder = H264Encoder()
    output = FfmpegOutput(file_name)

    # Set resolution and frame rate
    video_config = picam2.create_video_configuration(
        main={"size": (x_resolution, y_resolution)}
        )
    picam2.configure(video_config)

    # Debugging to confirm resolution is properly set:
    # print(picam2.video_configuration)

    # Take video
    print("Starting video...")
    picam2.start_recording(encoder, output)
    time.sleep(num_seconds)

    picam2.stop_recording()
    print("Stopped video...")
    picam2.close()

    # Wait after each test
    time.sleep(3)


"""
Define execution of desired tests here:
"""


def run_tests():

    # Varying resolution with 30 fps and 10 sec
    test_camera_video(640, 480, 10, "640x480_vid.mp4")
    #test_camera_video(1280, 720, 10, "1280x720_vid.mp4")
    #test_camera_video(1920, 1080, 10, "1920x1080_vid.mp4")


if __name__ == '__main__':
    run_tests()
