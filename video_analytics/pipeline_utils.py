""" Helper functions and architectures for mod.py file
"""
import cv2 
import os
import time
import numpy as np
import collections

from PIL import Image

CANNY_DEFAULT = 5
CANNY_LOW = 100
CANNY_HIGH = 200


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


def get_frame_feature(frame, 
                      edge_blur_rad=CANNY_DEFAULT, 
                      edge_blur_var=1, 
                      edge_canny_low=CANNY_LOW, 
                      edge_canny_high=CANNY_HIGH):
    """
    Gets edge detections in image using CV2. Taken from Reducto
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (edge_blur_rad, edge_blur_rad), edge_blur_var)
    edge = cv2.Canny(blur, edge_canny_low, edge_canny_high)
    return edge


def cal_frame_diff(edge, 
                   prev_edge, 
                   edge_thresh_low_bound):
    """
    Gets edge detections in image using CV2. Taken from Reducto
    """
    total_pixels = edge.shape[0] * edge.shape[1]
    frame_diff = cv2.absdiff(edge, prev_edge)
    frame_diff = cv2.threshold(frame_diff, edge_thresh_low_bound, 255, cv2.THRESH_BINARY)[1]
    changed_pixels = cv2.countNonZero(frame_diff)
    fraction_changed = changed_pixels / total_pixels
    return fraction_changed


def get_diff(curr, prev, var):
    """
    Gets frame difference factor

    Args:
        curr (_type_): _description_
        prev (_type_): _description_
        var (_type_): _description_

    Returns:
        _type_: _description_
    """
    edge_prev = get_frame_feature(prev, CANNY_DEFAULT, var, CANNY_LOW, CANNY_HIGH)
    edge_curr = get_frame_feature(curr, CANNY_DEFAULT, var, CANNY_LOW, CANNY_HIGH)

    diff = cal_frame_diff(edge_curr, edge_prev, 100)
    
    return diff


def cropped_detection(model, frame_in, frame_out):
    """
    Returns transformed output of cropped region from yolov5 model

    Args:
        model (_type_): _description_
        frame (_type_): _description_
    """
    edge = get_frame_feature(frame_in, CANNY_DEFAULT, 1, CANNY_LOW, CANNY_HIGH)
    x,y,w,h = cv2.boundingRect(edge)
    
    cropped = frame_out[y:(y + h), x:(x + w)]

    output = model(cropped).pandas().xywh[0]
    output['xcenter'] += x
    output['ycenter'] += y

    return output


def process_frame_diff(frame, prev, thresh) -> bool:
    """
    Determines whether or not to process a given frame
    """
    frame_var = np.var(frame)
    if get_diff(frame, prev, frame_var) > thresh:
        return True
    else: 
        return False

    
# Compute the frame difference
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


def ensure_gray(frame):
    # If frame is not grayscale
    if len(frame.shape) > 2 and frame.shape[2] > 1:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else: # if frame is already grayscale
        return frame


def draw_boxes(self, frame, model_out):
        """ Takes model's outputs (bounding box coordinates) and draws onto frame
            Returns the annotated frame 
        """
        color = (0,0,255)
        # Loops through all detections in frame
        for _, row in model_out.iterrows():
            x_center, y_center, width, height = row['xcenter'], row['ycenter'], row['width'], row['height']

            # Get top left corner coordinates
            topLeft = (int(x_center - width/2), int(y_center - height/2))
            bottomRight = (int(x_center + width/2), int(y_center + height/2))

            # Draw bounding box
            cv2.rectangle(frame, topLeft, bottomRight, color, thickness=2)

            # Add class name label
            cv2.putText(frame, str(row['name']), (topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
