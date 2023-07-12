import cv2 
import os
import time
import numpy as np

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


def process_frame_diff(frame, prev) -> bool:
    """
    Determines whether or not to process a given frame

    Args:
        frame (_type_): _description_
        prev (_type_): _description_

    Returns:
        bool: _description_
    """
    frame_var = np.var(frame)
    if get_diff(frame, prev, frame_var) > 0.005:
        return True
    else: 
        return False
    
    
def process_frame_diff_alternate(frame, frame_no, queue):
    if len(queue) == 3:
        return False
        
    if frame_no % 3 == 0:
        queue.append(frame)
        queue.pop(0)
        return True
    else: 
        return False

    