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

def draw_boxes(boxes, labels, image):
    """
    Draws the bounding box around a detected object.
    """
    for i, box in enumerate(boxes):
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            (255, 0, 0), 2
        )
        cv2.putText(image, str(labels[i]), (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2,
                    lineType=cv2.LINE_AA)
    return image






    