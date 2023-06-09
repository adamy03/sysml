import os
import sys
import argparse
import cv2
import torch
import time
import pandas as pd
import subprocess

from pathlib import Path
from process import *

# Set up path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
INFERENCE_PATH = '~/sysml/testing/test_results/temp.csv'
INPUT_FPS = 25

    
def run(
        yolov5_model,
        video_source,
        img_width,
        img_height,
        fps,          # TODO:no implementation yet
        max_frames,
        conf
        ):
    """
    Runs object detection pipeline given a model and video. 
    Returns runtime, number of frames, model outputs
    """

    # Setup for inference ----------------------------------------------------
    model = torch.hub.load('ultralytics/yolov5', yolov5_model)
    model.conf = conf  # NMS confidence threshold
    model.max_det = 100  # maximum number of detections per image

    # VIDEO ANALYSIS  --------------------------------------------------------
    # Read video, initialize output array, and being frame counter
    cap = cv2.VideoCapture(video_source)
    outputs = []

    # Get first
    ret, frame = cap.read()
    prev_out = model(frame, size=(img_width, img_height)).pandas().xywh[0]
    frames_processed = 1
    frame_no = 2
    
    # Start timer
    start = time.time()
    while frame_no <= max_frames:
        ret, frame = cap.read()

        if not ret:
            print('No frame returned')
            break

        if frame_no % (int(INPUT_FPS/fps)) == 0:
            output = model(frame, size=(img_width, img_height))
            output = output.pandas().xywh[0]
            output['frame'] = frame_no
            
            prev_out = output
            frames_processed += 1
            outputs.append(output)
        else:
            prev_out = prev_out.copy()
            prev_out['frame'] = frame_no
            outputs.append(prev_out)

        # prev_frame = frame
        frame_no += 1
        
    cap.release()
    end = time.time()
    
    # Process outputs --------------------------------------------------------
    runtime = end - start
    frames = frame_no - 1

    outputs = pd.concat(outputs)

    try:
        outputs.to_csv(INFERENCE_PATH)
    except:
        print('save failed')
        return -1

    print(
        f'frames: {frames}\n'
        + f'frames processed: {frames_processed}\n'
        + f'runtime (inference): {runtime}\n'
        + f'average time per frame: {runtime / frames}\n'
        + f'confidence: {conf}'
    , file=sys.stdout)

    return 1
    

"""
Parses the arguments into variables, for new logic simply add a new argument
Look through yolov5/detect.py for guidance on adding new arguments
"""
def parse_opt(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolov5-model', type=str, default='yolov5n', help='yolov5 model size')
    parser.add_argument('--video-source', type=str, default='sparse', help='input video path') 
    parser.add_argument('--img-width', type=int, default=1280, help='inference size width')
    parser.add_argument('--img-height', type=int, default=720, help='inference size height')
    parser.add_argument('--fps', type=int, default=250, help='frames to process per second of the video')
    parser.add_argument('--max-frames', type=int, default=250, help='max number of frames to process')
    parser.add_argument('--conf', type=float, default=0.6, help='model confidence threshold')
    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
