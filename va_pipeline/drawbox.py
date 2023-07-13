import os
import sys
import argparse
import cv2
import pandas as pd

from pathlib import Path
from process import *

def draw_boxes(video_path, ground_box, inference_box, out_path):
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    resX, resY, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and output video file
    #print(out_path)
    output_video = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (resX, resY))

    # Selected colors
    blue = (255, 0, 0)
    green = (0, 255, 0)
    
    dataframes = [(ground_box, green)]
    if inference_box is not None:
        dataframes.append((inference_box, blue))
        #print('hi')

    frameCount = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or cv2.waitKey(1) & 0xFF == ord('q'): break

        for df, color in dataframes:
            #print(dataframes)
            currRow = df[df['frame'] == frameCount] 
            for _, row in currRow.iterrows():
                x_center, y_center, width, height = row['xcenter'], row['ycenter'], row['width'], row['height']

                # Get top left corner coordinates
                topLeft = (int(x_center - width/2), int(y_center - height/2))
                bottomRight = (int(x_center + width/2), int(y_center + height/2))

                # Draw bounding box
                cv2.rectangle(frame, topLeft, bottomRight, color, 2)

                # Add class name label
                cv2.putText(frame, str(row['name']), (topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        output_video.write(frame)
        cv2.imshow('Current Frame', frame) 
        frameCount += 1

    cap.release()
    output_video.release()
    cv2.destroyAllWindows()


# # Data Selection/ Output
def run(
        inference_box,
        ground_box,
        video_source,
        out_path,
        all_videos,
        folder_vid,
        folder_csv,
        folder_dest
        ):
    
    if all_videos:
        print('hi')
        for csv in os.listdir(folder_csv):
           csv_name = csv
           csv_dir = folder_csv + '/' + csv
           csv_dir = csv_dir.replace("\\",'/')
           print(csv_dir)
           csv = pd.read_csv(csv_dir)
           for vid in os.listdir(folder_vid):
            # Checks to see if we have the correct video-csv pair
                if vid[0:-4] in csv_name:
                   vid_dir= folder_vid + vid
                   #print(f'../testing/test_results/model_comparison_videos/{csv[0:-4]}.mp4')
                   draw_boxes(vid_dir, csv, None, folder_dest + csv_name + '.mp4')
    else:
        # Formats all directory strings with forward slashes
        for i, item in enumerate([inference_box, ground_box, video_source, out_path]):
            if item is not None:
                item = item.replace('\\', '/')
                # Assign the modified string back to the original variable
                if i == 0:
                    inference_box = item
                elif i == 1:
                    ground_box = item
                elif i == 2:
                    video_source = item
                elif i == 3:
                    out_path = item

        

        # Prevents error from an unread csv and from single bounding box entry

        if ground_box != None and inference_box != None:
            ground_box = pd.read_csv(ground_box)
            inference_box = pd.read_csv(inference_box)

            draw_boxes(video_source, ground_box, inference_box, out_path)
        elif ground_box == None:
            inference_box = pd.read_csv(inference_box)

            draw_boxes(video_source, None, inference_box, out_path)
        elif inference_box == None:
            ground_box = pd.read_csv(ground_box)

            draw_boxes(video_source, ground_box, None, out_path)
        
        


"""
Parses the arguments into variables, for new logic simply add a new argument
Look through yolov5/detect.py for guidance on adding new arguments
"""
def parse_opt(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--inference-box', type=str, default=None, help='input inference_box.csv path')
    parser.add_argument('--ground-box', type=str, default=None, help='input ground_box.csv path')
    parser.add_argument('--video-source', type=str, default=None, help='input video path') 
    parser.add_argument('--out-path', type=str, default='../samples/testing/output_video.mp4', help='output folder location')
    parser.add_argument('--all-videos', type=bool, default= False, help='Turn on and give --folder-video, --folder-csv, and --folder-dest to automatically run on an entire dataset')
    parser.add_argument('--folder-vid', type=str, default=None, help='points towards folder of video names')
    parser.add_argument('--folder-csv', type=str, default='None', help="points towards folder of csv's" )
    parser.add_argument('--folder-dest', type=str, default=None, help="points to output video destination")
    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
