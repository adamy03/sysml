import cv2 
import os
import time
import numpy as np

from PIL import Image

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


def crop_region(
        img: np.array,
        box: tuple     
    ) -> np.array:
    
    return box[box[0]:box[1], box[2]:box[3]]


def get_frame_feature(frame, edge_blur_rad, edge_blur_var, edge_canny_low, edge_canny_high):
    """
    Gets edge detections in image using CV2. Taken from Reducto
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (edge_blur_rad, edge_blur_rad), edge_blur_var)
    edge = cv2.Canny(blur, edge_canny_low, edge_canny_high)
    return edge


def cal_frame_diff(edge, prev_edge, edge_thresh_low_bound):
    """
    Gets edge detections in image using CV2. Taken from Reducto
    """
    total_pixels = edge.shape[0] * edge.shape[1]
    frame_diff = cv2.absdiff(edge, prev_edge)
    frame_diff = cv2.threshold(frame_diff, edge_thresh_low_bound, 255, cv2.THRESH_BINARY)[1]
    changed_pixels = cv2.countNonZero(frame_diff)
    fraction_changed = changed_pixels / total_pixels
    return fraction_changed







    