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


def take_video(x_resolution, y_resolution, fps, num_seconds, file_name):
    # Initialize camera
    print("Initializing camera...")
    picam2 = Picamera2()

    # Initialize video properties
    encoder = H264Encoder()
    output = FfmpegOutput(file_name)

    # Set resolution and/or fps
    frame_dur = int(1.0 / fps * 1000000)
    video_config = picam2.create_video_configuration(
        main={"size": (x_resolution, y_resolution)},
        controls={"FrameDurationLimits": (frame_dur, frame_dur)}
        )
    # UNCOMMENT JUST the line starting "controls=" if you want to set fps
    
    picam2.configure(video_config)
    
    # Debugging to confirm the settings are correct
    print(picam2.video_configuration)

    # Take video
    print("Starting video...")
    picam2.start_recording(encoder, output)
    time.sleep(num_seconds)

    picam2.stop_recording()
    print("Stopped video...")
    picam2.close()


"""
Define execution of desired tests here:
"""


def run_tests():

    # Varying resolution with 30 fps and 10 sec
    #take_video(640, 480, 30, 10, "640x480_vid.mp4")
    #take_video(1280, 720, 30, 10, "1280x720_vid.mp4")
    #take_video(1920, 1080, 30, 10, "1920x1080_vid.mp4")
    
    # Varying fps with 1280x720 and 10 sec
    #take_video(1280, 720, 30, 10, "30fps_vid.mp4")
    #take_video(1280, 720, 15, 10, "15fps_vid.mp4")
    take_video(1280, 720, 1, 10, "1fps_vid.mp4")


if __name__ == '__main__':
    run_tests()
