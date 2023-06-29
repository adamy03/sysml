"""
This file takes the results (labels & bounding boxes) of running a Yolov5 model
on a video, and compares it to the results of running ground truth
(Yolov5 large on every frame of the video). It calculates accuracy using mAP.
"""

from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch
import pandas as pd


"""
Returns a list of dictionaries, one dictionary per frame that contains the
bounding box coordinates and labels of the ground truth model output.
"""
def get_ground_truth_list(fname):
    gt_list = []
    df = pd.read_csv(fname, sep=',')
    
    # Loop through the frame numbers in df
    num_frames = df.iloc[-1]['frame']
    for i in range(1, num_frames+1):
        
        # Filter df for just the rows corresponding to current frame
        current_frame_df = df[df['frame'] == i]
        current_frame_df = current_frame_df.sort_values(by=['class', 'xcenter'], ascending=True)
        
        #if not current_frame_df.empty:
        # Create FloatTensor containing boxes
        bbox_cols = current_frame_df[['xcenter', 'ycenter', 'width', 'height']]
        boxes = torch.tensor(bbox_cols.values)
        
        # Create IntTensor containing labels
        labels = torch.tensor(current_frame_df['class'].tolist())
        
        # Append dict with boxes, labels, and scores to the list
        frame_dict = {'boxes':boxes,
                    'labels':labels,
                    }
        gt_list.append(frame_dict)
            
        # If filtered df is empty, then there were no detections; append empty dict
        #else:
        #    gt_list.append({})
        
    return gt_list


"""
Returns a list of dictionaries, one dictionary per frame that contains the
bounding box coordinates, labels, and scores of the model prediction.
"""
def get_predictions_list(fname):
    preds_list = []
    df = pd.read_csv(fname, sep=',')

    # Loop through the frame numbers in df
    num_frames = df.iloc[-1]['frame']
    for i in range(1, num_frames+1):
        
        # Filter df for just the rows corresponding to current frame
        current_frame_df = df[df['frame'] == i]
        current_frame_df = current_frame_df.sort_values(by=['class', 'xcenter'], ascending=True)

        
        #if not current_frame_df.empty:
        # Create FloatTensor containing boxes
        bbox_cols = current_frame_df[['xcenter', 'ycenter', 'width', 'height']]
        boxes = torch.tensor(bbox_cols.values)
        
        # Create IntTensor containing labels
        labels = torch.tensor(current_frame_df['class'].tolist())
        
        # Create FloatTensor containing scores
        scores = torch.tensor(current_frame_df['confidence'].tolist())
        
        # Append dict with boxes, labels, and scores to the list
        frame_dict = {'boxes':boxes,
                    'labels':labels,
                    'scores':scores
                    }
        preds_list.append(frame_dict)
        
        # If filtered df is empty, then there were no detections; append empty dict
        #else:
        #    preds_list.append({})
        
    return preds_list

     
"""
Calculates mAP (Mean Average Precision) metric given ground truth and predictions
"""
def calculate_accuracy(ground_truth, prediction):
    metric = MeanAveragePrecision(box_format='xywh', iou_type="bbox")
    metric.update(prediction, ground_truth)
    result = metric.compute()
    return result['map'].item()


if __name__ == '__main__':
    
    # SPARSE: get lists for ground truth and predictions
    ground_truth = get_ground_truth_list("~/sysml/testing/test_results/config_testing/sparse/sparse_yolov5l_ground_truth_inference.csv")
    preds1280 = get_predictions_list("~/sysml/testing/test_results/config_testing/sparse/sparse_yolov5n_1280_720_25fps_inference.csv")
    #preds1536 = get_predictions_list("~/sysml/testing/test_results/config_testing/sparse/sparse_yolov5n_1536_864_25fps_inference.csv")

    # Calculate mAP scores
    print("mAP 1280: ", calculate_accuracy(ground_truth, preds1280))
    #print("mAP 1536: ", calculate_accuracy(ground_truth, preds1536))
    
    # MEDIUM: get lists for ground truth and predictions
    #ground_truth = get_ground_truth_list("~/sysml/testing/test_results/config_testing/medium/medium_yolov5l_ground_truth_inference.csv")
    #preds1280 = get_predictions_list("~/sysml/testing/test_results/config_testing/medium/medium_yolov5n_1280_720_25fps_inference.csv")
    #preds1536 = get_predictions_list("~/sysml/testing/test_results/config_testing/medium/medium_yolov5n_1536_864_25fps_inference.csv")
    
    # NOISY: get lists for ground truth and predictions
    #ground_truth = get_ground_truth_list("~/sysml/testing/test_results/config_testing/noisy/noisy_yolov5l_ground_truth_inference.csv")
    #preds1280 = get_predictions_list("~/sysml/testing/test_results/config_testing/noisy/noisy_yolov5n_1280_720_25fps_inference.csv")
    #preds1536 = get_predictions_list("~/sysml/testing/test_results/config_testing/noisy/noisy_yolov5n_1536_864_25fps_inference.csv")
