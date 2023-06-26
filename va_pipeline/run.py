import cv2
import torch
import pandas as pd
from PIL import Image
import time


print('start')
# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

cap = cv2.VideoCapture('../samples/NY_sample.mp4')
outputs = []
frame_no = 1

start = time.time()
ret, frame = cap.read()
while True:

        ret, frame = cap.read()
        if ret:
                out = model(frame)
                inf = out.pandas().xywh[0]
                inf['frame'] = frame_no
                outputs.append(inf)
                print(out)
        else:
                break
        frame_no += 1
end = time.time()

print(f'runtime: {end - start}')
print(f'frames: {frame_no - 1}')
df = pd.concat(outputs)
df.to_csv('../testing/test_results/yolov5s_NY.csv')
