import os
import sys
import cv2
import argparse
import collections

import numpy as np



# We need different modes
# - SIMPLE IN OUT MODE
# - : --video-source, --model 

# - TESTING
# - : 

# Outputs a cv2 frame
frame = Video(args)

#
frame = Analyze(frame, args)
frame = Preprocess(frame)



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

""" summary: allows us to store prev_frame curr_frame and next_frame for analysis
"""
class FrameQueue: 
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

    def fill_frames(self, cap):
        for i in range(3):
            ret, frame = cap.read()
            self.append(frame)

""" 
For reading video into our Frame Queue
"""
class Video:
    
    def __init__(
        self: cv2.VideoCapture, 
        cap,
        framerate =25,
        resolution =(1280, 720),
        fill_frames = True, # If analysis is not run, fills using previous analyzed frame bounding boxes
        ) -> None:
        
        if resolution is None:
            print('Using default Video Resolution')
            resolution = (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                )

        self.cap = cap
        self.framerate = framerate
        self.x_res = resolution[0]
        self.y_res = resolution[1]
        self.fill_frames = fill_frames

        self.frame_number = 1
        self.frame_queue = frame_queue
        
        return None


    def read_frame(self):
        """ summary: returns a single read frame and advances VideoCapture object
            returns: ret, frame 
        """ 
        return self.cap.read()
    
    def close_cap(self):
        """ summary: closes the cap object
        """ 
        self.cap.release()

    def fps_append(self, frame_number: int, fps: int, frame_queue: FrameQueue):
        """ summary: Only appends a specific number of frames per second to the cache
            returns: None if no more frames left
        """
        
        if frame_number % (int(self.framerate/fps)) == 0:
            ret, frame = self.cap.read()
            self.frame_number += 1
            if not ret:
                print(f"No frame returned from {self}")
                return None # I want to use this return None to signal ending the processing
            else:
                self.frame_number += 1
                frame_queue.append(frame)
        else:
            ret = self.cap.grab()
            if not ret:
                print('No frame returned')
                return None 
            self.frame_number += 1
            
            
""" 
LowLevelDecider: a class that contains multiple functions for determining whether to analyze a frame of video.s
Functions: frame differencing, background subtraction
Returns: True or False as to whether we run a model, along with model config (future)
"""
class LowLevelDecider:
    
    def __init__(
        self,
        frame_differencing: bool,
        frame_queue: FrameQueue,
    ):
        
        def ensure_gray(frame):
            # If frame is not grayscale
            if len(frame.shape) > 2 and frame.shape[2] > 1:
                return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else: # if frame is already grayscale
                return frame

        def frame_diff(prev_frame, cur_frame, next_frame):

            # Convert input frames to grayscale if not already grayscale
            prev_frame = ensure_gray(prev_frame)
            cur_frame = ensure_gray(cur_frame)
            next_frame = ensure_gray(next_frame)

            # Absolute difference between current frame and next frame
            diff_frames1 = cv2.absdiff(next_frame, cur_frame)

            # Absolute difference between current frame and # previous frame
            diff_frames2 = cv2.absdiff(cur_frame, prev_frame)

            # Return the result of bitwise 'AND' between the # above two resultant images
            # Then sum all of the pixel strengths
            diff_frame_out = cv2.bitwise_and(diff_frames1, diff_frames2)
            sum_pixels = np.sum(cv2.split(diff_frame_out))
            return sum_pixels



        if frame_differencing:
            frame_diff() # we want to have frame_diff return true or false, and 
        return
                



"""
AnalyzeFrame: 
Functions: frame differencing 
"""
class AnalyzeFrame:

    def __init__():
        return
    

"""
Preprocess: a class that contains multiple functions for preprocessing a single frame of video.
Each function returns the processed frame.
Functions: compress, crop
"""
class Preprocess:

    def __init__():
        return

    def compress(
            img: np.array,
            scaleX: float, 
            scaleY: float
        ) -> np.array:
        """
        Compresses an image given a x and y scale factor
        """    
        width = img.shape[0]
        height = img.shape[1]
        
        resized = cv2.resize(img, (int(width * scaleX), int(width * scaleY)))
        return resized

"""
Architecture of model within pipeline
"""
class Model():
    
    def __init__(
        self,
        model
        ) -> None:
        self.model = model
    
    def run_inference():
        print('I am here to stop errors!')
        
            
def RENAME_ME(
    video_path,
    model
):
    video = Video(cv2.VideoCapture(video_path))

    while True:
        ret, frame = video.read_frame()
        
        