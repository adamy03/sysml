import cv2
import torch
import time
import os
import sys
import pandas as pd
from PIL import Image
import time

OUT_WIDTH = 1920
OUT_HEIGHT = 1080
IN_WIDTH = 1920
IN_HEIGHT = 1080
WRITE_OUT = False
FRAME_CAP = 5

def run_pipeline(
        model = torch.hub.load('ultralytics/yolov5', 'yolov5n'), 
        video_path = './sysml/samples/medium.mp4'
        ):
    """
    Runs object detection pipeline given a model and video. 
    Returns runtime, number of frames, model outputs

    """

    # Read video, initialize output array, and being frame counter
    cap = cv2.VideoCapture(video_path)
    outputs = []
    frame_no = 1

    # Test if video was read
    ret, frame = cap.read()
    if not ret:
        raise ValueError('Could not read file')
    
    # Start timer
    start = time.time()
    while frame_no < FRAME_CAP:

        # Read frame
        ret, frame = cap.read()
        if ret:
            out = model(frame, size=[IN_WIDTH, IN_HEIGHT])
            inf = out.pandas().xywh[0]
            inf['frame'] = frame_no
            outputs.append(inf)
        
        if frame_no % 50 == 0:
            print(frame_no)

        frame_no += 1

    end = time.time()

    return end - start, frame_no - 1, pd.concat(outputs)

if __name__ == '__main__':
    runtime, frames, outputs = run_pipeline()

    outputs.to_csv('./sysml/testing/test_results/temp.csv')
    print(
        f'frames: {frames}\n' + 
        f'runtime: {runtime}\n' +
        f'average time per frame: {runtime / frames}'
    ) 
    