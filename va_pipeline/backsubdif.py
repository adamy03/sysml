from __future__ import print_function
import cv2 as cv2
import argparse

# Capture the frame from  video source
def get_frame(cap):
    # Capture the frame
    ret, frame = cap.read()

    if frame is None:
        return None

    # Resize the image
    #frame = cv2.resize(frame, None, fx=scaling_factor,
     #       fy=scaling_factor, interpolation=cv2.INTER_AREA)

    # Return the grayscale image
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

def frame_diff(prev_frame, cur_frame, next_frame):
    # Absolute difference between current frame and next frame
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
    args = parser.parse_args()

    ## [create]
    #create Background Subtractor objects
    if args.algo == 'MOG2':
        backSub = cv2.createBackgroundSubtractorMOG2()
    else:
        backSub = cv2.createBackgroundSubtractorKNN()
    ## [create]

    ## [capture]
    cap = cv2.VideoCapture(cv2.samples.findFileOrKeep(args.input))
    if not cap.isOpened():
        print('Unable to open: ' + args.input)

    ## [capture]




    prev_frame = backSub.apply(get_frame(cap))
    cur_frame = backSub.apply(get_frame(cap))
    next_frame = backSub.apply(get_frame(cap))

    while True:
        # Display the result of frame differencing
        cv2.imshow("Object Movement", frame_diff(prev_frame, cur_frame, next_frame))

        # Update the variables
        prev_frame = cur_frame
        cur_frame = next_frame
        next_frame = get_frame(cap)
        if next_frame is None:
            break
        #update the background model
        next_frame = backSub.apply(get_frame(cap))


        ## [display_frame_number]
        #get the frame number and write it on the current frame
        cv2.rectangle(cur_frame, (10, 2), (100,20), (255,255,255), -1)
        cv2.putText(cur_frame, str(cap.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
        ## [display_frame_number]

        ## [show]
        #show the current frame and the fg masks
        #cv2.imshow('Frame', frame)
        cv2.imshow('FG Mask', cur_frame)
        ## [show]

        keyboard = cv2.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break
    cv2.destroyAllWindows()