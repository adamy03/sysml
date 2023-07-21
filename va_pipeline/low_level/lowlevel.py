from __future__ import print_function
import cv2 as cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Compute the frame difference
def frame_diff(prev_frame, cur_frame, next_frame):
    # Absolute difference between current frame and next frame
    diff_frames1 = cv2.absdiff(next_frame, cur_frame)

    # Absolute difference between current frame and # previous frame
    diff_frames2 = cv2.absdiff(cur_frame, prev_frame)

    # Return the result of bitwise 'AND' between the # above two resultant images
    return cv2.bitwise_and(diff_frames1, diff_frames2)

# Capture the frame from webcam/video Input
def get_frame(capture):
    # Capture the frame
    ret, frame = capture.read()
    if ret == True:
        gray_frame = backSub.apply(frame)
    # Resize the image
    #frame = cv2.resize(frame, None, fx=scaling_factor,
     #       fy=scaling_factor, interpolation=cv2.INTER_AREA)

    # Return the grayscale image
        return gray_frame, frame
    else: 
        return None, None
    # return  # THIS MIGHT BE UNECESSARY

parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
parser.add_argument('--output', type=str, help='Type in outputfile path and file name.', default=None)
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
parser.add_argument('--fdif', action='store_true', help='Turns on frame differencing after background subtraction ')
parser.add_argument('--capframe', type=int, help='Type the frame # you want to capture', default=None)
parser.add_argument('--compcapture', nargs='+', type=int, help='Type your desired compressed resolution (960, 540)', default=None)
args = parser.parse_args()

## [create]
#create Background Subtractor objects
if args.algo == 'MOG2':
    backSub = cv2.createBackgroundSubtractorMOG2()
else:
    backSub = cv2.createBackgroundSubtractorKNN()

# create an empty DataFrame to store the frame number and Pixel sum
df = pd.DataFrame(columns=['Frame', 'Pixel Sum'])
## [create]

## [capture]
capture = cv2.VideoCapture(cv2.samples.findFileOrKeep(args.input))
if not capture.isOpened():
    print('Unable to open: ' + args.input)
    exit(0)
## [capture]

## [write]
# We obtain resolutions from the frame, we convert resolutions from float to integer.
frame_width = int(capture.get(3))
frame_height = int(capture.get(4))
fps = capture.get(cv2.CAP_PROP_FPS)
print(f'FPS: {fps}')

# Define codec and create VideoWriter object. 
if args.compcapture == None:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (frame_width, frame_height))
## [write]

if args.fdif == True:
    # Runs Background subtraction on first 3 frames
    prev_frame, _ = get_frame(capture)
    cur_frame, _ = get_frame(capture)
    next_frame, _ = get_frame(capture)

while True:
    if args.fdif == True:  # Frame Differncing on Back Subtraction
        # Update the variables
        prev_frame = cur_frame
        cur_frame  = next_frame
        next_frame, frame = get_frame(capture)  # Returns background subtracted new frame

        if next_frame is None:
            print("No more frames to process")
            break
        ## [apply]
        output_frame = frame_diff(prev_frame, cur_frame, next_frame)
        output_frame_color = cv2.cvtColor(output_frame, cv2.COLOR_GRAY2BGR)
        ## [apply]

    if args.compcapture != None:
        ret, frame = capture.read()
        if frame is None:
            print("No more frames to process")
            break
        print(args.compcapture)
        output_frame_color = cv2.resize(frame, args.compcapture)


    else:                  # Default Back Subtraction
        ret, frame = capture.read()
        if frame is None:
            print("No more frames to process")
            break

        ## [apply]
        #update the background model
        output_frame = backSub.apply(frame)
        output_frame_color = cv2.cvtColor(output_frame, cv2.COLOR_GRAY2BGR)
        ## [apply]

    ## [display_frame_number]
    #get the frame number and write it on the current frame
    cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv2.putText(frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    ## [display_frame_number]



    ## [write data]
    if args.capframe != None: # Checks if capframe argument was included
        if capture.get(cv2.CAP_PROP_POS_FRAMES) == args.capframe:
            for i in range(5): # Num frames skipped
                cv2.imwrite(f'capture{i}.jpg', frame)
                cv2.imwrite(f'processed_capture{i}.jpg', output_frame_color)
                for i in range(5):
                    ret, frame = capture.read()
                    if frame is None:
                        print("No more frames to process")
                        break
            capture.release()
    if args.compcapture == None:
        out.write(output_frame_color)  # Video Output 
        
        b = np.sum(cv2.split(output_frame)) # Sums all pixel values of output frame
        df = df.append({'Frame': capture.get(cv2.CAP_PROP_POS_FRAMES), 'Pixel Sum': b}, ignore_index=True) 
    ## [write data]

    ## [show]
    #show the current frame and the fg masks
    cv2.imshow('Frame', frame)
    cv2.imshow('FG Mask', output_frame_color)
    ## [show]

    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break

# When all done, release video capture and video write objects
capture.release()

if args.compcapture == None:
    out.release()
print('all cv2 objects are released')

# Write Collected data
df.to_csv('pixel_data.csv', index=False)

# Close all the frames
cv2.destroyAllWindows()
