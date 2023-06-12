import cv2 
import os
import time
import numpy as np
from PIL import Image

def test_compress(file_path, out_path, scaleX, scaleY, out_fps):
    cap = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_width = int(frame_width * scaleX)
    out_height = int(frame_width * scaleY)

    print('Input dimensions: ({}, {})'.format(frame_width, frame_height))

    out = cv2.VideoWriter(
        out_path, 
        cv2.VideoWriter_fourcc(*'mp4v'), 
        out_fps,
        (out_width, out_height)
        )

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:  # no more frames to read
            break
        
        resized = cv2.resize(frame, (out_width, out_height))
        out.write(resized)

    cap.release()
    out.release()

    return

    
if __name__ == '__main__':
    test_compress(
        './samples/intersection_video.mp4', 
        './samples/compressed_intersection.mp4',
        0.1,
        0.1,
        60
        )
        