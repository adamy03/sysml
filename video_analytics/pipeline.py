import os
import sys
import argparse
import cv2
import torch
import time
import pandas as pd
import subprocess

from pathlib import Path
from pipeline_utils import *
import collections

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
        conf,
        differencing_toggle
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
    if not cap.isOpened():
        print('Unable to open: ' + video_source)
        exit(0)
    outputs = []

    # Instantiates and fills our queue
    frame_queue = FrameQueue() 
    frame_queue.fill_frames(cap) 

    # Gets current and processes it (skips first frame)
    prev_out = model(frame_queue.get_current(), size=(img_width, img_height)).pandas().xywh[0]
    frames_processed = 1
    frame_no = 2
    
    # Start timer
    start = time.time()
    while frame_no <= max_frames:

        if frame_no % (int(INPUT_FPS/fps)) == 0:
            ret, frame = cap.read()
            if not ret:
                print('No frame returned')
                break
            # add valid frame to queue
            frame_queue.append(frame)

            # TODO: We now perform frame differencing to determine whether we run the model on the current 
            # frame (and probably the next few frames)
            # - We can choose to run frame differencing more often than our model to improve reaction time
            # - We want to be able to process the next x frames

            if differencing_toggle:

                sum_pixels = frame_diff(frame_queue.get_previous(), frame_queue.get_current(), frame_queue.get_next())
                print(sum_pixels)
                cv2.imshow("Video Proccessing", frame)
                keyboard = cv2.waitKey(30)
                if keyboard == 'q' or keyboard == 27:
                    break
                
                if sum_pixels > 3500000: # TODO: Automatically determining this threshold is our next area of research
                    print("processed")
                    output = model(frame_queue.get_current(), size=(img_width, img_height))
                    output = output.pandas().xywh[0]
                    output['frame'] = frame_no
                    frames_processed += 1
                    outputs.append(output)
                    prev_out = output
                else:
                    prev_out = prev_out.copy()
                    prev_out['frame'] = frame_no
                    print("appended previous frame")
                    outputs.append(prev_out)
            else:
                # Regularly run model if frame differencing is not toggled
                cv2.imshow("Video Proccessing", frame)
                output = model(frame_queue.get_current(), size=(img_width, img_height))
                output = output.pandas().xywh[0]
                output['frame'] = frame_no
                frames_processed += 1
                outputs.append(output)
                prev_out = output
            
        else:
            ret, frame = cap.read()
            if not ret:
                print('No frame returned')
                break
            prev_out = prev_out.copy()
            prev_out['frame'] = frame_no
            print("appended previous frame")
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
    parser.add_argument('--fps', type=int, default=25, help='frames to process per second of the video')
    parser.add_argument('--max-frames', type=int, default=250, help='max number of frames to process')
    parser.add_argument('--conf', type=float, default=0.5, help='model confidence threshold')
    parser.add_argument('--differencing-toggle', action='store_true', help='turns on frame-differencing energy saving')
    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
