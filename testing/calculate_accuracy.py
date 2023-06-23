"""
This file takes the results (labels & bounding boxes) of running a Yolov5 model
on a video, and compares it to the results of running ground truth
(Yolov5 large on every frame of the video). It calculates accuracy using mAP.
"""

from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch
import pandas as pd

# Assumes file is run from testing directory
FILE_PATH = "~/sysml/testing/model_test/yolov5-on-rpi4-2020/johnston_yolov5/yolov5/runs/detect"
ground_truth = []


def get_predictions_list(file_location, num_files):
    predictions_list = []
    # Loop through the files containing labels + bboxes for each frame
    for i in range(1, num_files+1):
        fname = file_location + "_" + str(i) + ".txt"
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
        # scores = torch.tensor(df['conf-score'])
        
        # Append dict with boxes, labels, and scores to the list
        frame_dict = {'boxes':boxes,
                      'labels':labels,
                      # 'scores':scores
                      }
        predictions_list.append(frame_dict)
        
        #if (i%50 == 0):
            #print(predictions_list)
        
    return predictions_list
        

def calculate_accuracy(ground_truth, prediction):
    metric = MeanAveragePrecision(iou_type="bbox")
    metric.update(prediction, ground_truth)
    result = metric.compute()
    return result['map'].item()


if __name__ == '__main__':
    ground_truth = get_predictions_list(FILE_PATH + "/sparse_yolov5l_gt/labels/sparse", 200)
    print(ground_truth)
    
    #mAP = calculate_accuracy(ground_truth, predictions)
    #print("mAP: ", mAP)
