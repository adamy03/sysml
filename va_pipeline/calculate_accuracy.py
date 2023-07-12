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
def get_ground_truth_list(width, height, fname, num_frames):
    gt_list = []
    df = pd.read_csv(fname, sep=',')
    
    # Normalize values
    df['xcenter'] /= width
    df['ycenter'] /= height
        
    
    # Loop through the frame numbers in df
    for i in range(1, num_frames+1):
        
        # Filter df for just the rows corresponding to current frame
        current_frame_df = df[df['frame'] == i]
        current_frame_df = current_frame_df.sort_values(by=['class', 'xcenter'], ascending=True)
        
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
        
    return gt_list


"""
Returns a list of dictionaries, one dictionary per frame that contains the
bounding box coordinates, labels, and scores of the model prediction.
"""
def get_predictions_list(width, height, fname, num_frames):
    preds_list = []
    df = pd.read_csv(fname, sep=',')
    
    if df.empty:
        df = df.astype('float64')
    
    # Normalize values
    df['xcenter'] /= width
    df['ycenter'] /= height

    # Loop through the frame numbers in df
    for i in range(1, num_frames+1):
        
        # Filter df for just the rows corresponding to current frame
        current_frame_df = df[df['frame'] == i]
        current_frame_df = current_frame_df.sort_values(by=['class', 'xcenter'], ascending=True)

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
        
    return preds_list

     
"""
Calculates mAP (Mean Average Precision) metric given ground truth and predictions
"""
def calculate_accuracy(ground_truth, prediction):
    metric = MeanAveragePrecision(box_format='cxcywh', iou_type="bbox")
    metric.update(prediction, ground_truth)
    result = metric.compute()
    return result['map'].item()


"""
Calculate mAP for different vids & resolutions of testing suite; append to df
"""
def mAP_resolution(df, vid_names, vid_sizes):
    for source in vid_names:
        for size in vid_sizes:
            res_width = size[0]
            res_height = size[1]
            
            # Get ground truth list
            gt = get_ground_truth_list(1280, 720, '~/sysml/testing/test_results/config_testing/resolution/' + 
                                       f'{source}_1280_720_25_inference.csv',
                                    200)

            # Get predictions list
            preds = get_predictions_list(res_width, res_height, '~/sysml/testing/test_results/config_testing/resolution/' + 
                                        f'{source}_{res_width}_{res_height}_25_inference.csv',
                                        200)

            mAP = calculate_accuracy(gt, preds)
            
            # Add new row to dataframe
            new_row = {'Video': source, 'Width': res_width, 'Height': res_height, 'Framerate': 25, 'mAP': mAP}
            df.loc[len(df)] = new_row
            
            # Write mAP to file
            file_dir = f'C:/Users/shiva/sysml/testing/test_results/config_testing/resolution/' + f'{source}_{res_width}_{res_height}_25_stats.txt'

            with open(file_dir, 'a') as f:
                f.write(f'\nmAP: {mAP}\n')
            
            
    return df

"""
Calculate mAP for different vids & frame rates of testing suite; append to df
"""
def mAP_framerate(df, vid_names, frame_rates):
    for source in vid_names:
        for fps in frame_rates:
            # Get ground truth list
            gt = get_ground_truth_list(1280, 720, '~/sysml/testing/test_results/config_testing/resolution/' + 
                                       f'{source}_1280_720_25_inference.csv',
                                    200)

            # Get predictions list
            preds = get_predictions_list(1280, 720, '~/sysml/testing/test_results/config_testing/framerate/' + 
                                        f'{source}_1280_720_{fps}_inference.csv',
                                        200)

            mAP = calculate_accuracy(gt, preds)
            
            # Add new row to dataframe
            new_row = {'Video': source, 'Width': 1280, 'Height': 720, 'Framerate': fps, 'mAP': mAP}
            df.loc[len(df)] = new_row
            
            # Write mAP to file
            file_dir = f'C:/Users/shiva/sysml/testing/test_results/config_testing/framerate/' + f'{source}_1280_720_{fps}_stats.txt'

            with open(file_dir, 'a') as f:
                f.write(f'\nmAP: {mAP}\n')
    return df


if __name__ == '__main__':
    
    model = 'yolov5n'
    framerate = 25
    frame_cap = 250
    conf = 0.6
    
    # Set up dataframe
    cols = ['Video', 'Width', 'Height', 'Framerate', 'mAP']
    df = pd.DataFrame(columns=cols)
    
    # Calculate mAP for diff vids & resolutions
    #vid_names = ['largefast', 'largeslow', 'smallfast', 'smallslow']
    vid_names = ['largefast']
    vid_sizes = [[1280, 720], [960, 540], [640, 360]]
    df = mAP_resolution(df, vid_names, vid_sizes)
    
    print(df)

    # Calculate mAP for diff vids & frame rates
    frame_rates = [1, 3, 5]
    df = mAP_framerate(df, vid_names, frame_rates)

    print(df)
    #df.to_csv("mAPscores.csv")
    
    # Print mAP scores
    #mAP = calculate_accuracy(gt, preds)
    #print(f"{source}, {model}, {conf}, {res_width}, {res_height}")
    #print("mAP: ", " ", mAP)
