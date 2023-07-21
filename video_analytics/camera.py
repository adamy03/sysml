import time
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput


"""
Takes a video num_seconds long at the specified resolution, frame rate,
and output file name (must end in .mp4).
"""


def take_video(x_resolution: int, y_resolution: int, fps: int, num_seconds: int, file_name):
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
    # print(picam2.video_configuration)

    # Take video
    print("Starting video...")
    picam2.start_recording(encoder, output)
    time.sleep(num_seconds)

    picam2.stop_recording()
    print("Stopped video...")
    picam2.close()


"""
Takes 'no_images' number of images at the specified resolution.
"""


def take_image(x_resolution: int, y_resolution: int, no_images: int, save: bool = False):
    picamera = Picamera2()
    config = picamera.create_still_configuration(
        main={"size": (x_resolution, y_resolution)}
    )
    picamera.configure(config)

    picamera.start()
    for i in range(no_images):
        time.sleep(2)
        if save:
            picamera.capture_file("{}x{}.jpg".format(x_resolution, y_resolution))
        else:
            picamera.capture_array()

    picamera.stop
    picamera.close()
    return
