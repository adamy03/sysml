import cv2
import torch
from PIL import Image

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')

cap = cv2.VideoCapture('../../../samples/NY_sample.mp4')
ret, frame = cap.read()

while cap.isOpened():
        out = model(frame)
        ret, frame = cap.read()

        print(out.pandas().xywh[0])
