import cv2
import torch
import pandas as pd
import time


# Load model
model = torch.hub.load('ultralytics/yolov5', 'yolov5l')

# Open video source
cap = cv2.VideoCapture('../videos/medium.mp4')
outputs = []
frame_no = 1

# Loop through video frames
start = time.time()
ret, frame = cap.read()
while True:
        ret, frame = cap.read()
        if ret and frame_no <= 200:
                out = model(frame)
                inf = out.pandas().xywh[0]
                inf['frame'] = frame_no
                outputs.append(inf)
                print(out)
                print(f'frame: {frame_no}')
        else:
                break
        frame_no += 1
end = time.time()

# Print stats and save results
print(f'runtime: {end - start}')
print(f'frames: {frame_no - 1}')
df = pd.concat(outputs)
print(df.shape)
df.to_csv('../testing/test_results/yolov5l_medium.csv')
