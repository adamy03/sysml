import os
import sys
import argparse
import cv2
import pandas as pd

from pathlib import Path
from process import *

def draw_boxes(video_path, ground_box, inference_box, out_path):
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    resX, resY, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and output video file
    output_video = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (resX, resY))

    # Selected colors
    blue = (255, 0, 0)
    green = (0, 255, 0)
    
    dataframes = [(ground_box, green)]
    if inference_box is not None:
        dataframes.append((inference_box, blue))

    frameCount = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or cv2.waitKey(1) & 0xFF == ord('q'): break

        for df, color in dataframes:
            currRow = df[df['frame'] == frameCount] 
            for _, row in currRow.iterrows():
                x_center, y_center, width, height = row['xcenter'], row['ycenter'], row['width'], row['height']

                # Get top left corner coordinates
                topLeft = (int(x_center - width/2), int(y_center - height/2))
                bottomRight = (int(x_center + width/2), int(y_center + height/2))

                # Draw bounding box
                cv2.rectangle(frame, topLeft, bottomRight, color, 2)

                # Add class name label
                cv2.putText(frame, str(row['name']), (topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        output_video.write(frame)
        cv2.imshow('Current Frame', frame) 
        frameCount += 1

    cap.release()
    output_video.release()
    cv2.destroyAllWindows()


# # Data Selection/ Output

def run(
        inference_box,
        ground_box,
        video_path,
        output_path
        ):
    inference_box = inference_box.replace('\\','/')
    ground_box = ground_box.replace('\\','/')
    video_path = video_path.replace('\\','/')
    output_path = output_path.replace('\\','/')

    if inference_box == None:
        draw_boxes(video_path, ground_box, inference_box, output_path)
    else:
        draw_boxes(video_path, ground_box, None, output_path)

    


"""
Parses the arguments into variables, for new logic simply add a new argument
Look through yolov5/detect.py for guidance on adding new arguments
"""
def parse_opt(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--inference-box', type=str, default=None, help='input inference_box.csv path')
    parser.add_argument('--ground_box', type=str, default=None, help='input ground_box.csv path')
    parser.add_argument('--video-source', type=str, default=None, help='input video path') 
    parser.add_argument('--out-path', type=str, default='../samples/testing/output_video.mp4', help='output folder location')
    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
