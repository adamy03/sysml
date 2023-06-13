import time
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput

class Camera():
    def __init__(self):
        self.camera = Picamera2()


"""
Takes a video num_seconds long at the specified resolution, frame rate,
and output file name (must end in .mp4).
"""


def take_video(x_resolution: int, y_resolution: int, fps: int, num_seconds: int, file_name):
    # Initialize video properties
    encoder = H264Encoder()
    output = FfmpegOutput(file_name)

    # Set resolution and/or fps
    frame_dur = int(1.0 / fps * 1000000)
    video_config = camera.create_video_configuration(
        main={"size": (x_resolution, y_resolution)},
        controls={"FrameDurationLimits": (frame_dur, frame_dur)}
        )
    # UNCOMMENT JUST the line starting "controls=" if you want to set fps
    
    camera.configure(video_config)
    
    # Debugging to confirm the settings are correct
    # print(camera.video_configuration)

    # Take video
    print("Starting video...")
    camera.start_recording(encoder, output)
    time.sleep(num_seconds)

    camera.stop_recording()
    print("Stopped video...")
    camera.close()


"""
Takes 'no_images' number of images at the specified resolution.
"""


def take_image(x_resolution: int, y_resolution: int, no_images: int, save: bool = False):
    config = self.camera.create_still_configuration(
        main={"size": (x_resolution, y_resolution)}
    )
    self.camera.configure(config)

    self.camera.start()
    for i in range(no_images):
        time.sleep(2)
        if save:
            self.camera.capture_file("{}x{}.jpg".format(x_resolution, y_resolution))
        else:
            self.camera.capture_array()

    self.camera.stop
    self.camera.close()
    return
