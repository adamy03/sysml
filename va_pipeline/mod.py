import argparse
import cv2
import torch
import pandas as pd
import os
import sys
from pathlib import Path
import time

# Set up path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


# Define constants
OUT_WIDTH = 1920
OUT_HEIGHT = 1080
WRITE_OUT = False
FRAME_CAP = 251


"""
Runs object detection pipeline given a model and video. 
Returns runtime, number of frames, model outputs
"""
def run(
        yolov5_model,
        video_source,
        img_size,
        fps,          # no implementation yet
        ):
    
    # Setup for inference ----------------------------------------------------

    width = img_size[0]
    height = img_size[1]

    model = torch.hub.load('ultralytics/yolov5', yolov5_model)


    # VIDEO ANALYSIS  --------------------------------------------------------
    # Read video, initialize output array, and being frame counter
    cap = cv2.VideoCapture(video_source)
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
            out = model(frame, size=[width, height])
            inf = out.pandas().xywh[0]
            inf['frame'] = frame_no
            outputs.append(inf)
        
        if frame_no % 50 == 0:
            print(frame_no)

        frame_no += 1

    end = time.time()
    
    
    # Process outputs --------------------------------------------------------
    runtime = end - start
    frames = frame_no - 1
    outputs = pd.concat(outputs)

    outputs.to_csv('./sysml/testing/test_results/temp.csv')
    print(
        f'frames: {frames}\n' + 
        f'runtime (inference): {runtime}\n' +
        f'average time per frame: {runtime / frames}'
    )


"""
Parses the arguments into variables, for new logic simply add a new arugment
Look through yolov5/detect.py for guidance on adding new arguments
"""
def parse_opt(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolov5-model', type=str, default='yolov5n.pt', help='yolov5 model name')
    parser.add_argument('--video-source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--img-size', nargs='+', default=[1920, 1080], help='inference size [w,h]')
    parser.add_argument('--fps', type=int, default=25, help='frames to process per second of video')
    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
