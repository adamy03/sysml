"""
This file tests pi energy usage for camera while:
    - idle
    - taking one photo
    - taking multiple photos
"""
import time
import os
from PIL import Image
from picamera2 import Picamera2

"""
Testing camera while idle. Wait time in seconds
"""
def test_camera_idle(x_resolution: int, y_resolution: int, wait_time: int):
    picamera = Picamera2()
    config = picamera.create_video_configuration(
        main={"size": (x_resolution, y_resolution)}
    )
    picamera.configure(config)
    
    picamera.start()
    
    print("waiting %i seconds..." % (wait_time))
    time.sleep(wait_time)
    
    picamera.stop()
    picamera.close()
    return


"""
Testing camera energy over series of images.
"""
def test_camera_image(x_resolution: int, y_resolution: int, no_images: int, save: bool = False):
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

"""
Define execution of desired tests here:
"""
if __name__ == '__main__':
    start = time.time()
    test_camera_idle(1920, 1080, 5)
    print("pi start to end: {}".format(time.time()-start))
