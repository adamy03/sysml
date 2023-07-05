import os
import sys
import argparse
import cv2
import torch
import time
import pandas as pd

from pathlib import Path
from process import *

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
#INFERENCE_PATH = './sysml/testing/test_results/temp.csv'
INFERENCE_PATH = '~/sysml/testing/test_results/mAP_experiments/0.5_conf/noisy_yolov5n_640_360_0.5conf.csv'

def process_frame(frame, prev) -> bool:
    frame_var = np.var(frame)
    if get_diff(frame, prev, frame_var) > 0.005:
        return True
    else: 
        return False
    
def run(
        yolov5_model,
        video_source,
        img_width,
        img_height,
        fps,          # TODO:no implementation yet
        frame_cap,
        conf
        ):
    """
    Runs object detection pipeline given a model and video. 
    Returns runtime, number of frames, model outputs
    """
    # Regular Inf Path
    INFERENCE_PATH = f'~/sysml/testing/test_results/mAP_experiments/{conf}_conf/{video_source}_{yolov5_model}_{img_width}_{img_height}_{conf}conf.csv'
    
    # Ground Truth Inf Path
    #INFERENCE_PATH = f'~/sysml/testing/test_results/mAP_experiments/{conf}_conf/{video_source}_{yolov5_model}_ground_truth_{conf}conf.csv'
    
    # Setup for inference ----------------------------------------------------
    model = torch.hub.load('ultralytics/yolov5', yolov5_model)
    model.conf = conf  # NMS confidence threshold
    model.max_det = 100  # maximum number of detections per image

    # VIDEO ANALYSIS  --------------------------------------------------------
    # Read video, initialize output array, and being frame counter
    cap = cv2.VideoCapture(f'../../sysml/samples/{video_source}.mp4') # Remember to change to './sysml/samples/sparse.mp4' for pi usage
    outputs = []

    # Test if video was read
    ret, frame = cap.read()
    if not ret:
        raise ValueError('Could not read file')
    
    # Get inital readings
    prev_frame = frame
    out = model(prev_frame, size=[img_width, img_height])
    prev_inf = out.pandas().xywh[0]
    prev_inf['frame'] = 1
    outputs.append(prev_inf.copy())
    frame_no = 2

    # Start timer
    start = time.time()
    while frame_no <= frame_cap:
        ret, frame = cap.read()
        if not ret:
            print('No frame returned')
            break

        if process_frame(frame, prev_frame):
            out = model(frame, size=[img_width, img_height])
            inf = out.pandas().xywh[0]

            inf['frame'] = frame_no
            outputs.append(inf)
            prev_inf = inf.copy()
        else:
            prev_inf['frame'] = frame_no
            outputs.append(prev_inf.copy())

        frame_no += 1
        prev_frame = frame
        
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
        outputs.to_csv('temp.csv')

    print(
        f'frames: {frames}\n' + 
        f'runtime (inference): {runtime}\n' +
        f'average time per frame: {runtime / frames}'
    )
    

"""
Parses the arguments into variables, for new logic simply add a new argument
Look through yolov5/detect.py for guidance on adding new arguments
"""
def parse_opt(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolov5-model', type=str, default='yolov5n', help='yolov5 model size')
    parser.add_argument('--video-source', type=str, default=f'medium', help='input video path') 
    parser.add_argument('--img-width', type=int, default=1280, help='inference size width')
    parser.add_argument('--img-height', type=int, default=720, help='inference size height')
    parser.add_argument('--fps', type=int, default=25, help='frames to process per second of the video')
    parser.add_argument('--frame-cap', type=int, default=250, help='max number of frames to process')
    parser.add_argument('--conf', type=float, default=0.4, help='model confidence threshold')
    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
