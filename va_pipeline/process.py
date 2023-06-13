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





    