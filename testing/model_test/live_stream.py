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

XRES = 640
YRES = 480
lowres = (XRES, YRES)
FPS = 15

####################################################################
# CAMERA SET UP #
####################################################################

picam2 = Picamera2()

config = picam2.create_preview_configuration(main={'size': (1280, 720), 'format': 'RGB888'})
lowres_config = picam2.create_preview_configuration(main={'size': (640, 480), 'format':'RGB888'})
picam2.configure(config)
picam2.set_controls({"FrameRate": FPS})


####################################################################
#  VIDEO #
####################################################################

out_path = './test_video.mp4'
out_fps = FPS
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
yolo_net = cv2.dnn.readNet('./models/yolov5s.onnx')
class_list = []
with open("./models/classes.txt", "r") as f:
    class_list = [cname.strip() for cname in f.readlines()]

# Define the torchvision image transforms
def format_yolov5(frame):
    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame

    return result

def yolo_infer(image):
    input_image = format_yolov5(image) # making the image square
    blob = cv2.dnn.blobFromImage(input_image , 1/255.0, (640, 640))
    yolo_net.setInput(blob)
    
    outputs = yolo_net.forward()
    output_data = outputs[0]

    class_ids = []
    confidences = []
    boxes = []

    image_width, image_height, _ = input_image.shape
    x_factor = image_width / 640
    y_factor =  image_height / 640

    for r in range(25200):
        row = output_data[r]
        confidence = row[4]
        if confidence >= 0.4:

            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > .25):

                confidences.append(confidence)

                class_ids.append(class_id)

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45) 

    result_class_ids = []
    result_confidences = []
    result_boxes = []

    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])

    for i in range(len(result_class_ids)):

        box = result_boxes[i]
        class_id = result_class_ids[i]

        cv2.rectangle(image, box, (0, 255, 255), 2)
        cv2.rectangle(image, (box[0], box[1] - 20), (box[0] + box[2], box[1]), (0, 255, 255), -1)
        cv2.putText(image, class_list[class_id], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0))
    
    return image


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
    # if frames % 10 == 0:
    #     yolo_infer(image)

    out.write(image)
    frames += 1
    
    elapsed_time = time.time() - start
    if elapsed_time < frame_delay:
        time.sleep(frame_delay - elapsed_time)
    
    if frames == 100:
        picam2.switch_mode(lowres_config)
    if frames > 200:
        break

print(frames)
picam2.stop
picam2.close()
    
out.release()
