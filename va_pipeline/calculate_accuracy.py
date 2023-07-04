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
def get_ground_truth_list(width, height, fname):
    gt_list = []
    df = pd.read_csv(fname, sep=',')
    
    # Normalize values
    df['xcenter'] /= width
    df['ycenter'] /= height
    
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
def get_predictions_list(width, height, fname):
    preds_list = []
    df = pd.read_csv(fname, sep=',')
    
    # Normalize values
    df['xcenter'] /= width
    df['ycenter'] /= height

    # Loop through the frame numbers in df
    num_frames = df.iloc[-1]['frame']
    for i in range(1, int(num_frames+1)):
        
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
        frame_dict = {'boxes': boxes,
                    'labels': labels,
                    'scores': scores
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


# if __name__ == '__main__':

    # Change to name and path of files
    res_width = 1280
    res_height = 720
    model = 'yolov5n'
    source = 'noisy'
    framerate = 25
    frame_cap = 250
    save_results = True
    
    # Get ground truth list
    gt_path = f'~/sysml/testing/test_results/config_testing/{source}/{source}_yolov5l_ground_truth.csv'
    gt = get_ground_truth_list(1920, 1080, gt_path)

    # Get preds list
    pred_dir = f'~/sysml/testing/test_results/config_testing/{source}/'
    pred_name = f'{source}_{model}_{res_width}_{res_height}_{framerate}fps'
    pred_path = pred_dir + pred_name + '_inference.csv'
    preds = get_predictions_list(res_width, res_height, pred_path)

    # Calculate mAP scores
    mAP = calculate_accuracy(gt, preds)
    print("mAP: ", res_width, " ", mAP)

    # Write mAP score to file
    file_dir = f'C:/Users/Chelsey/sysml/testing/test_results/config_testing/{source}/'
    file_path = file_dir + pred_name + '_stats.txt'

    # Open the file
    with open(file_path, 'a') as f:
        f.write(f'\nmAP: {mAP}\n')
        # Write more content as needed
