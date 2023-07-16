from __future__ import print_function
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Capture the frame from  video source
def get_frame(cap, writer=None):
    # Capture the frame
    ret, frame = cap.read()

    if ret == True and writer != None:
        # Write the frame into the output file
        video_writer.write(cap)

    if frame is None:
        return None

    # Resize the image
    #frame = cv2.resize(frame, None, fx=scaling_factor,
     #       fy=scaling_factor, interpolation=cv2.INTER_AREA)

    # Return the grayscale image
    return ret, cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

def frame_diff(prev_frame, cur_frame, next_frame):
    # Absolute difference between current frame and next frame
    print(next_frame)
    print(cur_frame)
    diff_frames1 = cv2.absdiff(next_frame, cur_frame)

    # Absolute difference between current frame and # previous frame
    diff_frames2 = cv2.absdiff(cur_frame, prev_frame)

    # Return the result of bitwise 'AND' between the # above two resultant images
    return cv2.bitwise_and(diff_frames1, diff_frames2)





if __name__=='__main__':
    parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
    parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
    parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
    parser.add_argument('--output', type=str, help='Type in outputfile path and file name.', default=None)
    args = parser.parse_args()

    ## [create]
    #create Background Subtractor objects
    if args.algo == 'MOG2':
        backSub = cv2.createBackgroundSubtractorMOG2()
    else:
        backSub = cv2.createBackgroundSubtractorKNN()
    ## [create]

    # Create an empty DataFrame to store the frame number and RGB sum
    df = pd.DataFrame(columns=['Frame', 'Pixel Sum'])
    
    ## [capture]
    cap = cv2.VideoCapture(cv2.samples.findFileOrKeep(args.input))
    if not cap.isOpened():
        print('Unable to open: ' + args.input)

    ## Video Writing 
    output_file = args.output
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    video_writer = cv2.VideoWriter(output_file, fourcc, 25, (1280, 720))
    
    ## [capture]
    prev_frame = backSub.apply(get_frame(cap)[1])
    cur_frame = backSub.apply(get_frame(cap)[1])
    next_frame = backSub.apply(get_frame(cap)[1])

    frame_number = 0
    while True:
        # Display the result of frame differencing
        

        # Update variables, increment frames
        prev_frame = cur_frame
        cur_frame = next_frame
        ret, next_frame = get_frame(cap)
        if next_frame is None:
            break

        #update the background model
        next_frame = backSub.apply(next_frame)
        if ret == True:
            video_writer.write(next_frame)

        ## [display_frame_number]
        #get the frame number and write it on the current frame
        cv2.rectangle(cur_frame, (10, 2), (100,20), (255,255,255), -1)
        cv2.putText(cur_frame, str(cap.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
        ## [display_frame_number]

        ## [show]
        #show the current frame and the fg masks
        #cv2.imshow('Frame', frame)
        #processed_frame = frame_diff(prev_frame, cur_frame, next_frame)
        #cv2.imshow("Object Movement", processed_frame)
        cv2.imshow('FG Mask', next_frame)
        # Write the manipulated frame to the output video
        ## [show]

        b = np.sum(cv2.split(cur_frame))
        df = df.append({'Frame': frame_number, 'Pixel Sum': b}, ignore_index=True)

        keyboard = cv2.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break
        #print(frame_number)
        frame_number +=1

    cap.release()
    video_writer.release()
    print(df)
    df.to_csv('rgb_data.csv', index=False)
    cv2.destroyAllWindows()