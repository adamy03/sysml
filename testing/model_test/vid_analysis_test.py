import torch
import torchvision
import torchvision.transforms as transforms
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import cv2
import time
import numpy as np
from PIL import Image

"""
This file opens a video and runs the SSD300 VGG16 model on each frame to draw
and label bounding boxes for object detection.
"""


"""
Function to run a single image through model and get boxes, labels, and scores
"""


def predict(image, model, detection_threshold):
    # transform the image to tensor
    image = transform(image)

    # add a batch dimension
    image = image.unsqueeze(0) 
    
    image = image.to('cuda')

    # get the predictions on the image
    outputs = model(image) 

    # get score for all the predicted objects
    pred_scores = outputs[0]['scores'].to('cpu')
    pred_scores = pred_scores.detach()

    # get all the predicted bounding boxes and filter by threshold
    pred_bboxes = outputs[0]['boxes'].to('cpu')
    pred_bboxes = pred_bboxes.detach()
    boxes = pred_bboxes[pred_scores >= detection_threshold]

    # get all predicted labels and filter by threshold    
    labels = outputs[0]['labels'].to('cpu')
    labels = labels[pred_scores >= detection_threshold]

    scores = pred_scores[pred_scores >= detection_threshold]

    return boxes, labels, scores 


"""
Read in a video and loop through its frames using the OpenCV Library.
Runs predict() function on each frame
"""


def should_process_frame(frame, prev_frame, index):
  return True

def process_video(video_path, model):
        model = model.to('cuda')
        ground_truth = {}
        comparison = {}

        # Set up video capture
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cv2.VideoCapture.get(cap, int(cv2.CAP_PROP_FRAME_COUNT)))
        print(f'Frame count: {frame_count}')
        ret, frame = cap.read()
        prev_frame = None

        index = -1
        start_time = time.time()

        # Store the results of the last processed frame. If we skip a frame, fill "comparison" with the 
        # last processed frame
        prev_frame_results = (None, None, None)

        # Loop through frames
        while cap.isOpened():
            if not ret:
                break
            index += 1
            if index > 1:
              break
            if index%100 == 0:
              print(f'Reached index {index}')
            image_id = f'image{index}'
            ground_truth[image_id] = {}
            comparison[image_id] = {}

            # Run prediction on this frame. We have to run it regardless of our filtering
            # method so we can assess ground truth
            boxes, labels, scores = predict(frame, model, 0.3)
            print(boxes)
            
            ground_truth[image_id]['boxes'] = boxes
            ground_truth[image_id]['labels'] = labels

            if should_process_frame(frame, prev_frame, index):
                # This becomes the last frame processed
                prev_frame_results = (boxes, labels, scores)
                prev_frame = frame.copy()

                # Comparison contains results of this frame
                comparison[image_id]['boxes'] = boxes
                comparison[image_id]['labels'] = labels
                comparison[image_id]['scores'] = scores
            else:
                # Use the previous frame's results instead
                comparison[image_id]['boxes'] = prev_frame_results[0]
                comparison[image_id]['labels'] = prev_frame_results[1]
                comparison[image_id]['scores'] = prev_frame_results[2]
            ret, frame = cap.read()

        end_time = time.time()
        print(f'Total time: {end_time - start_time}')
        
        # Calculate % of frames selected and write their indexes to json file
        cap.release()

        ground_truth_formatted = [v for k,v in ground_truth.items()]
        comparison_formatted = [v for k,v in comparison.items()]

        # print(f'Ground truth: {ground_truth_formatted}')
        # print(f'Comparison: {comparison_formatted}')

        return ground_truth_formatted, comparison_formatted


"""
Function that takes "ground truth" boxes and labels and "comparison" boxes,
labels, and scores, and returns the accuracy of the comparison result relative
to the ground truth 
"""
def calculate_accuracy(ground_truth, prediction):
    metric = MeanAveragePrecision(iou_type="bbox")
    metric.update(prediction, ground_truth)
    result = metric.compute()
    return result['map'].item()


"""
Set up the model
"""

# Define the torchvision image transforms
transform = transforms.Compose([transforms.ToTensor(),])

# Load the object detection model, SSD, and set mode to eval
model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
model.eval()

"""
Run the video through the model and get results
"""
gt, comp = process_video('/home/pi/sysml/testing/sensing/video_test/1920x1080_vid.mp4', model)
mAP = calculate_accuracy(gt, comp)
print(mAP)
