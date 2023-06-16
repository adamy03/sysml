import cv2
import time
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from picamera2 import Picamera2

####################################################################
# SETUP #
####################################################################

CAM_ID = -1
XRES = 640
YRES = 480
lowres = (XRES, YRES)
FPS = 15

####################################################################
# CAMERA SET UP #
####################################################################

picam2 = Picamera2()

config = picam2.create_preview_configuration(main={"size": (XRES, YRES), "format": "BGR888"})
picam2.configure(config)
picam2.set_controls({"FrameRate": FPS})


####################################################################
#  VIDEO #
####################################################################

out_path = './test_video.mp4'
out_fps = 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(
    out_path,
    fourcc, 
    FPS,
    (XRES, YRES),
    isColor=True
    )

####################################################################
# INFERENCE #
####################################################################
net = cv2.dnn.readNet('./models/yolov5s.onnx')
with open('', 'r') as f:
   class_names = f.read().split('\n')

# Define the torchvision image transforms
def format_yolov5(frame):
    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame

    return result

####################################################################
# PIPELINE START #
####################################################################

frames = 0
frame_delay = 1 / FPS
exec_start = time.time()

print('starting model')
picam2.start()
while True:
    start = time.time()

    # Capture
    buffer = picam2.capture_array()
    image = cv2.resize(buffer, (XRES, YRES))

    # Inference
    if frames % 10 == 0:
        input_image = format_yolov5(image) # making the image square
        blob = cv2.dnn.blobFromImage(input_image , 1/255.0, (640, 640), swapRB=True)
        net.setInput(blob)
        outputs = net.forward()

    out.write(image)
    frames += 1
    
    # elapsed_time = time.time() - start
    # if elapsed_time < frame_delay:
    #     time.sleep(frame_delay - elapsed_time)
    
    if time.time() - exec_start > 7:
        break

print(frames)
picam2.stop
picam2.close()
    
out.release()
