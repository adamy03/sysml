import os
import sys
import cv2
import argparse
import collections
import numpy as np
import torch
import pandas as pd
from pipeline_utils import draw_boxes

"""
This file contains classes for execution of analysis using Yolov5 object detection models on videos.
Classes included: Video, FrameCache, Model
Accessed by [INSERTNAMEHERE******************] to run analysis testing
"""


""" 
FrameCache: a cache of frames to preprocess/analyze
"""
class FrameCache: 
    def __init__(
        self,
        max_frames=1,  # max number of frames allowed in the cache
        ):
        self.max_frames = max_frames
        self.frames = []  # a list of tuples: (frame_number, frame_data)


    def __str__(self) -> str:
        return str(self.frames)


    def add_frame(self, frame_num, frame):
        """ Adds frame to cache, popping the oldest frame
        """
        self.frames.append((frame_num, frame)) # tuple (number, frame data)
        if len(self.frames) > self.max_frames: # if we exceed max # of frames allowed in cache
            self.frames.pop(0)
    
    
    def get_frame(self, index=0):
        """ Returns copy of tuple (frame_num, frame_data) at index in the cache
            If index not specified, returns oldest frame in the cache (index=0)
        """
        frame_info = tuple(self.frames[index])
        return frame_info



""" 
Video: wrapper around cv2 Video Capture object that reads frames
"""
class Video:
    def __init__(
        self, 
        path:str=None,  # file path of video
        framerate=25,
        resolution=None,
        show_feed=False
        ) -> None:
    
        self.path = path
        self.framerate = framerate
        self.show_feed = show_feed

        # Define other variables
        self.cap = cv2.VideoCapture(path)  # Video Capture object to iterate through video frames
        if not self.cap.isOpened():
            raise ValueError('Could not read file path.')
        self.curr_frame_num = 1
        self.ret = True  # True until all frames are retrieved from video 

        # Use input video resolution if 'resolution' not specified
        if resolution is None:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f'Using default video resolution: {width}, {height}')
            resolution = (width, height)
        else:
            self.x_res = resolution[0]
            self.y_res = resolution[1]


    def __str__(self) -> str:
        return str(self.cap)


    def close_cap(self):
        """ Releases Video Capture object
        """ 
        self.cap.release()


    def read_frame(self):
        """ Returns ret, frame from the video
        """ 
        self.frame_number += 1
        return self.cap.read()


    def add_to_cache(self, frames_skip: int, frame_cache: FrameCache) -> None:
        """ Adds one frame to the cache, a FrameCache object
            Skips frames to emulate a specific fps reading
        """
        
        if self.curr_frame_num != 1:
            # Skip frames (to emulate fps) until we reach the one we wish to add
            frames_passed = 0
            while frames_passed < frames_skip and self.ret:
                self.ret = self.cap.grab()  # advances to next frame
                if self.ret:  # Frame was returned
                    self.curr_frame_num += 1
                    frames_passed += 1
                else:  # No frame returned
                    print(f'No frame returned from {self}')
    
        # Read frame and add to cache
        self.ret, frame = self.cap.read()
        if self.ret:  # Frame was returned
            frame_cache.add_frame(self.curr_frame_num, frame)  
            self.curr_frame_num += 1         
        else:  # No frame returned
            print(f'No frame returned from {self}')

        return frame_cache
    
    def fill_cache(self, frames_skip, frame_cache: FrameCache):
        """ Fills cache; called at the beginning of video analysis execution
        """
        # Loop and add frame to cache until cache is full
        for i in range(frame_cache.max_frames):
            self.add_to_cache(frames_skip, frame_cache)
        return frame_cache



"""
Model: wrapper around Yolov5 model for object detection
"""
class Model:
    
    def __init__(
        self,
        model: str,  # options: 'yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x'
        conf = 0.6,
        max_det = 100,
        as_json:bool=False  # Default: saves output bounding boxes as .csv; if True, saves as .json
        ) -> None:
        
        # Load Model from Ultralytics library
        self.model = torch.hub.load('ultralytics/yolov5', model)
        self.model.conf = conf  # NMS confidence threshold
        self.model.max_det = max_det  # maximum number of detections per image

        # Add settings for model outputs
        self.outputs = {}  # key: frame num; value: dataframe of model outputs
        self.prev_out = None  # dataframe of bounding box detections for most recent frame
        self.frames_processed = 0
        self.as_json = as_json


    def run(self, frame, video: Video):
        """ Runs model on one frame of video
            Returns detections (bounding boxes) as a dataframe
        """
        # Get output
        detections = self.model(frame)
        self.prev_out = detections  # to make detections for most recent frame accessible
        self.frames_processed += 1

        # Convert detections to readable dataframe and append to dict of outputs for video
        if self.as_json:
            detections = detections.json()
            self.outputs[video.curr_frame_num] = detections
        else:  # save as .csv
            detections = detections.pandas().xywh[0]
            detections['frame'] = video.curr_frame_num  # Add column to store frame # in dataframe
            self.outputs[video.curr_frame_num] = detections
            
        # Go back and fill in detections for any skipped frames 
        self.fill_prev_frames(video.curr_frame_num)
    
        return detections


    def fill_prev_frames(self, curr_frame_num):
        """ Fills in detections (in self.outputs) for any skipped frames by using the most
            recent bounding box detections.
            Ex. If model was run on frames 1 and 4, this function will fill in the detections
            for frames 2 and 3 using the detections for frame 1
        """
        last_frame_num = list(self.outputs.keys())[-1]
        frame_num_diff = curr_frame_num - last_frame_num

        if frame_num_diff < 1:
            return None

        # Loop through the missing frame numbers, filling in detections
        for i in range(frame_num_diff):
            frame_num = last_frame_num + i
            if self.as_json:
                self.outputs[frame_num] = self.prev_output
            else:  # store in .csv
                output = self.prev_out.copy()
                output['frame'] = frame_num  # Add column to store frame # in dataframe
                self.outputs[frame_num] = output


    def __shivani__function(): 
        print("MORE FUNCTIONS!!!")

 
"""
Framework for running test. 
"""   
# class Test:
#     def __init__(
#         self,
#         test_set:dict,
#         save_inference:bool=False,
#         inference_path:str='',
#         run_energy:bool=False,
#         energy_path:str=None
#         ) -> None:
        
#         self.test_set = test_set
#         self.save_inference = save_inference
#         self.inference_path = inference_path
#         self.run_energy = run_energy
#         self.energy_path = energy_path
        
#         if self.save_inference and not inference_path:
#             raise ValueError('No path for inference given.')
#         if self.run_energy and not energy_path:
#             raise ValueError('No path for energy given.')
        
#         return None
    
#     def run_test(self, as_csv, as_json):
#         for video in self.test_set.items():
#             cap = Video(video)
            

# Function Graveyard: to be revived


# check: ccan this do frame differencing, background subtraction


# video with frames
# 0 1 2 3 4 5 6

# fps: skip every other frame

# framenum = 0
# model_out = model(frame0)

# for 
# cache: 0 2 4
# for 

# cache: 2 4 6


# we get cache
# we do processornot on 2

# model out for 2


# processornot on 4
# do not process 4
# fill frame output for 4 with the output of 2