"""
This file tests pi energy usage for camera while:
    - idle
    - taking one photo
    - taking multiple photos
"""
import time
from picamera2 import Picamera2


"""
Testing camera while idle.
"""
def test_camera_idle(x_resolution, y_resolution, wait_time):
    # Initialize camera
    print("init camera...")
    picam2 = Picamera2()
    preview_config = picam2.create_preview_configuration(
        main={"size": (x_resolution, y_resolution)}
        )

    print("starting camera...")
    picam2.start()
    print("waiting %i seconds..." % (wait_time))
    time.sleep(wait_time)
    picam2.stop()

    return


"""
Testing camera energy over series of images.
"""
def test_camera_image(x_resolution, y_resolution, no_images):
    # Initialize camera
    print("init camera...")
    picam2 = Picamera2()
    preview_config = picam2.create_preview_configuration(
        main={"size": (x_resolution, y_resolution)}
        )

    print("starting camera...")
    picam2.start()

    for i in range(no_images):
        picam2.capture_array("main")

    picam2.stop()

    return

"""
Define desired tests here:
"""
def run_tests():
    test_camera_idle(600, 800, 10)


if __name__ == '__main__':
    run_tests()
