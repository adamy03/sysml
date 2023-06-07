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
    picamera = Picamera2()
    config = picamera.create_preview_configuration(
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
    config = picamera.create_preview_configuration(
        main={"size": (x_resolution, y_resolution)}
    )
    picamera.configure(config)
    
    picamera.start()
    for i in range(no_images):
        time.sleep(2)
        if save:
            picamera.start_and_capture_files(
                "{}x{}".format(x_resolution, y_resolution)+"test{:d}.jpg", 
                initial_delay=0, 
                delay=0, 
                num_files=1
                )
        else:
            picamera.capture_array


    picamera.stop()
    picamera.close()
    return

"""
Define execution of desired tests here:
"""
if __name__ == '__main__':
    print("64 x 64")
    test_camera_image(64, 64, 1, True)
    print("640 x 480")
    test_camera_image(640, 480, 1, True)
    print("1920 x 1080")
    test_camera_image(1920, 1080, 1, True)
