
#We have a series of frames in a list[[frame 1], [frame 2], [frame 3]]

#current frame = timerCount * fps


"""
This file takes the results (labels & bounding boxes) of running a Yolov5 model
on a video, and compares it to the results of running ground truth
(Yolov5 large on every frame of the video). It calculates accuracy using mAP.
"""

from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch
import pandas as pd
import os

# Assumes file is run from testing directory
FILE_PATH = "~/sysml/testing/model_test/yolov5-on-rpi4-2020/johnston_yolov5/yolov5/runs/detect"
ground_truth = []


"""
Returns a list of dictionaries, one dictionary per frame that contains the
bounding box coordinates and labels of the ground truth model output.
"""
def get_ground_truth_list(file_location, num_files):
    gt_list = []
    # Loop through the files containing labels + bboxes for each frame
    for i in range(1, num_files+1):
        fname = file_location + "_" + str(i) + ".txt"
        # Read data
        df = pd.read_csv(fname, sep=' ', header=None)
        cols = ['label', 'x-center', 'y-center', 'width', 'height']
        df.columns = cols
        
        # Create FloatTensor containing boxes
        bbox_cols = df[['x-center', 'y-center', 'width', 'height']]
        boxes = torch.tensor(bbox_cols.values, dtype=torch.float32)
        
        # Create IntTensor containing labels
        labels = torch.tensor(df['label'])
        
        # Append dict with boxes, labels, and scores to the list
        frame_dict = {'boxes':boxes,
                      'labels':labels,
                      }
        gt_list.append(frame_dict)
        
    return gt_list


"""
Returns a list of dictionaries, one dictionary per frame that contains the
bounding box coordinates, labels, and scores of the model prediction.
"""
def get_predictions_list(file_location, num_files):
    predictions_list = []
    # Loop through the files containing labels + bboxes for each frame and if there is no bounding box data we will use the previous

    # We will then start from frame1, and if 

    # We will loop through, if the file contents are empty then we will compare
    j=1
    # Create a set of existing bounding box files

    for i in range(1, num_files+1):

        fname = file_location + "_" + str(i) + ".txt"
        
        # checks whether we have written a bounding box for a given frame
        if not os.path.exists(fname):
            print(f"no label for frame {i}: filling with frame {j}" )
            fname = file_location + "_" + str(j) + ".txt"
        else:
            j=i

        # Read data
        df = pd.read_csv(fname, sep=' ', header=None)
        cols = ['label', 'x-center', 'y-center', 'width', 'height', 'conf-score']
        df.columns = cols
        
        # Create FloatTensor containing boxes
        bbox_cols = df[['x-center', 'y-center', 'width', 'height']]
        boxes = torch.tensor(bbox_cols.values, dtype=torch.float32)
        
        # Create IntTensor containing labels
        labels = torch.tensor(df['label'])
        
        # Create FloatTensor containing scores
        scores = torch.tensor(df['conf-score'])
        
        # Append dict with boxes, labels, and scores to the list
        frame_dict = {'boxes':boxes,
                      'labels':labels,
                      'scores':scores
                      }
        predictions_list.append(frame_dict)
        
    return predictions_list

     
"""
Calculates mAP (Mean Average Precision) metric given ground truth and predictions
"""
def calculate_accuracy(ground_truth, prediction):
    metric = MeanAveragePrecision(box_format='xywh', iou_type="bbox")
    metric.update(prediction, ground_truth)
    result = metric.compute()
    return result['map'].item()


if __name__ == '__main__':
    
    # Get lists for ground truth and predictions
    ground_truth = get_ground_truth_list(FILE_PATH + "/sparse_yolov5l_gt/labels/sparse", 200)
    preds = get_predictions_list(FILE_PATH + "/sparse_yolov5l_gt/labels/sparse", 200)
    
    # Calculate mAP score
    mAP = calculate_accuracy(ground_truth, preds)
    print("mAP: ", mAP)
