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
Classes: Video, FrameCache, Model
"""


""" 
FrameCache: a cache of frames to preprocess/analyze
"""
class FrameCache: 
    def __init__(
        self,
        max_frames=1,
        ):
        self.max_frames = max_frames
        self.frames = []


    def __str__(self) -> str:
        return str(self.frames)


    def add_frame(self, frame):
        """ Adds frame to cache, popping the oldest frame
        """
        self.frames.append(frame)
        if len(self.frames) > self.max_frames:
            self.frames.pop(0)
    
    
    def get_frame(self, index=0):
        """ Returns copy of frame at index in the cache
        """
        return self.frames[index].copy()
        


""" 
Video: wrapper around cv2 Video Capture object that reads frames
"""
class Video:
    def __init__(
        self, 
        cap:cv2.VideoCapture,
        cap_framerate=25,
        resolution=None,
        output_video_dir=None,
        show_output=False
        ) -> None:
        
        # Use input video resolution if 'resolution' not specified
        if resolution is None:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f'Using default video resolution: {width}, {height}')
            resolution = (width, height)
        
        # Set other parameters
        self.output_video_dir = output_video_dir
        self.cap = cap
        self.cap_framerate = cap_framerate
        self.ret = True

        self.x_res = resolution[0]
        self.y_res = resolution[1]

        self.frame_number = 1
        self.show_output = show_output

        if output_video_dir is not None: 
            # Creates a video writer for video outputs
            # TODO: Video writing will have to be called by 
            codec = codec = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(output_video_dir,
                                                codec,
                                                cap_framerate,
                                                resolution
                                                )


    def __str__(self) -> str:
        return str(self.cap)


    def close_cap(self):
        """ Releases video capture object
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
            Returns: None if no more frames
        """
        # Skip frames (to emulate fps) until we reach the one we wish to add
        frames_passed = 0
        while frames_passed < frames_skip:
            self.ret = self.cap.grab()  # advances to next frame
            if not self.ret:  # if self.ret is False
                print(f'No frame returned from {self}')
                self.frame_number += 1
            frames_passed += 1
    
        # Read frame and add to cache
        self.ret, frame = self.cap.read()

        # No frame was returned
        if not self.ret:
            print(f'No frame returned from {self}')
            
        # Frame was returned
        else:
            self.frame_number += 1
            frame_cache.add_frame(frame)


    def write_frame(self, frame):
        """ Writes a frame to Video object VideoWriter
        """
        self.video_writer.write(frame)
        if self.show_output:
            cv2.imshow('Writing Video', frame)
            # Press 'q' key to stop writing and close the display window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return None




"""
Model: wrapper around Yolov5 model for object detection
"""
class Model:
    
    def __init__(
        self,
        model: str,  # options: 'yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x'
        conf = 0.6,
        max_det = 100,
        as_json:bool=False
        ) -> None:
        
        # Load Model from Ultralytics library
        self.model = torch.hub.load('ultralytics/yolov5', model)
        self.model.conf = conf  # NMS confidence threshold
        self.model.max_det = max_det  # maximum number of detections per image

        # Settings for model outputs
        self.outputs = {}  # key: frame num; value: dataframe of model outputs
        self.prev_out = None
        self.frames_processed = 0
        self.as_json = as_json


    def run(self, frame, video: Video):
        """ Runs model on one frame of video
            Returns detections (bounding boxes) as a dataframe
        """
        detections = self.model(frame)
        self.frames_processed += 1

        # Convert output to readable dataframe
        if self.as_json:
            detections = detections.json()
            self.outputs[video.frame_number] = detections
        else:
            detections = detections.pandas().xywh[0]
            detections['frame'] = video.frame_number
            self.outputs[video.frame_number] = detections

        # Make output accessible to low level decider to fill in detections for skipped frames
        self.prev_out = detections

        # If we wish to save the video annotated with detections
        if video.output_video_dir != None:
            self.write_annotated_frame(self, frame, video, detections)
        
        return detections
    
    
    def write_annotated_frame(self, frame, video, detections):
        """ Annotates a frame with bounding box detections and writes frame to video
            Called by run()
        """
        annotated_frame = draw_boxes(frame, detections)  # Draws boxes onto frame
        video.write_frame(annotated_frame)  # Writes frame to Video Writer in Video() object

        if video.show_output: # Shows annotated frame
            cv2.imshow('Writing Video', frame)


    def fill_prev_frames(self, curr_frame_number):
        """ Fills in detections (in self.outputs) for any skipped frames by using the most
            recent bounding box detections.
            Ex. If model was run on frames 1 and 4, this function will fill in the detections
            for frames 2 and 3 using the detections for frame 1
        """
        last_frame_no = self.outputs.keys()[-1]
        frame_no_diff = curr_frame_number - last_frame_no

        if frame_no_diff < 1:
            return None

        # Loop through the missing frame numbers
        for i in range(frame_no_diff):
            curr_frame = last_frame_no + i
            if not self.as_json:
                output = self.prev_out.copy()
                output['frame'] = curr_frame
                self.outputs[curr_frame] = output
            else:
                self.outputs[curr_frame] = self.prev_output.copy()


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