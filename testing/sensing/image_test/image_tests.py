"""
This file tests pi energy usage for camera while:
    - idle
    - taking one photo
    - taking multiple photos
"""
import time

from picamera2 import Picamera2
from video_test.taking_video_test import init_camera


"""
Testing camera while idle.
"""
def test_camera_idle(x_resolution, y_resolution, wait_time):
    picam2 = Picamera2()
    preview_config = picam2.create_preview_configuration(
        main={"size": (x_resolution, y_resolution)}
        )
    picam2.start()
    
    print("waiting %i seconds..." % (wait_time))
    time.sleep(wait_time)
    
    picam2.stop()
    return


"""
Testing camera energy over series of images.
"""
def test_camera_image(x_resolution, y_resolution, no_images):
    picam2 = Picamera2()
    preview_config = picam2.create_preview_configuration(
        main={"size": (x_resolution, y_resolution)}
        )
    picam2.start()

    for i in range(no_images):
        time.sleep(2)
        print(i)
        picam2.capture_array("main")

    picam2.stop()
    return

"""
Define execution of desired tests here:
"""
if __name__ == '__main__':
    test_camera_image(64, 64, 5)
    test_camera_image(640, 480, 5)
    test_camera_image(1920, 1080, 5)
