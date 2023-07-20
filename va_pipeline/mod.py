import os
import sys
import argparse
import cv2
import torch
import time
import pandas as pd
import subprocess
import collections

from pathlib import Path
from process import *

class FrameQueue: # allows us to store previous current and next frames for analysis
    def __init__(self, max_frames=3):
        self.max_frames = max_frames
        self.frames = collections.deque(maxlen=max_frames)

    def append(self, frame):
        self.frames.append(frame)

    def get_previous(self):
        if len(self.frames) < 2:
            return None
        return self.frames[-2]

    def get_current(self):
        if len(self.frames) < 1:
            return None
        return self.frames[-1]

    def get_next(self):
        # This function should be used only after appending the next frame.
        if len(self.frames) < 3:
            return None
        return self.frames[0]

# Set up path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
FPS = 25 # Framerate of video
INFERENCE_PATH = '~/sysml/testing/test_results/temp.csv'

    
def run(
        yolov5_model,
        video_source,
        img_width,
        img_height,
        fps,          
        frame_cap,
        conf,
        video_path,
        out_path
        ):
    """
    Runs object detection pipeline given a model and video. 
    Returns runtime, number of frames, model outputs
    """

    # Regular Inf Path
    #INFERENCE_PATH = f'~/sysml/testing/test_results/mAP_experiments/{conf}_conf/{video_source}_{yolov5_model}_{img_width}_{img_height}_{conf}conf.csv'
    # INFERENCE_PATH = f'../testing/test_results/new_video_results/{video_source}_{yolov5_model}_{img_width}_{img_height}_{conf}conf.csv'
    if out_path == None:
        INFERENCE_PATH = f'~/sysml/samples/testing/ground_truth/test3/{video_source}_{yolov5_model}_{img_width}_{img_height}_{conf}.csv'
        print(INFERENCE_PATH)
    else:
        INFERENCE_PATH = f'{out_path}/{video_source}_{yolov5_model}_{img_width}_{img_height}_{conf}.csv'


    # Setup for inference ----------------------------------------------------
    model = torch.hub.load('ultralytics/yolov5', yolov5_model)
    model.conf = conf  # NMS confidence threshold
    model.max_det = 100  # maximum number of detections per image

    # VIDEO ANALYSIS  --------------------------------------------------------
    # Read video, initialize output array, and being frame counter
    if video_path == None:
        cap = cv2.VideoCapture(f'../samples/testing/videos/test2/{video_source}.mp4') # Remember to change to './sysml/samples/sparse.mp4' for pi usage
    else:
        cap = cv2.VideoCapture(f'{video_path}/{video_source}.mp4') # Remember to change to './sysml/samples/sparse.mp4' for pi usage
    #subprocess.run("cd", shell=True)
    # cap = cv2.VideoCapture(f'./sysml/samples/{video_source}.mp4') # Remember to change to './sysml/samples/sparse.mp4' for pi usage
    #cap = cv2.VideoCapture(f'./sysml/samples/{video_source}.mp4') # Remember to change to './sysml/samples/sparse.mp4' for pi usage
    
    # Get first
    outputs = []
    ret, frame = cap.read()
    prev_out = model(frame, size=(img_width, img_height)).pandas().xywh[0]
    prev_out['frame'] = 1
    
    outputs.append(prev_out)
    frames_processed = 1
    frame_no = 2
    
    # Start timer
    start = time.time()
    while frame_no <= frame_cap:
        ret, frame = cap.read()

        if not ret:
            print('No frame returned')
            break

        if frame_no % (int(FPS/fps)) == 0:
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
        pass

    print(
        f'frames: {frames}\n' + 
        f'runtime (inference): {runtime}\n' +
        f'average time per frame: {runtime / frames}\n' +
        f'confidence: {conf}'
    , file=sys.stdout)

    return outputs
    

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
    parser.add_argument('--fps', type=int, default=25, help='frames to process per second of the video')
    parser.add_argument('--frame-cap', type=int, default=250, help='max number of frames to process')
    parser.add_argument('--conf', type=float, default=0.6, help='model confidence threshold')
    parser.add_argument('--video-path', type=str, default=None, help='directory path for test videos')
    parser.add_argument('--out-path', type=str, default=None, help='output folder location')
    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
