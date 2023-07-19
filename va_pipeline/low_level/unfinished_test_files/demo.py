import cv2
import numpy as np
import torch

# Compute the frame difference
def frame_diff(prev_frame, cur_frame, next_frame):

    # Convert input frames to grayscale
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
    cur_frame = cv2.cvtColor(cur_frame, cv2.COLOR_RGB2GRAY)
    next_frame = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)

    # Absolute difference between current frame and next frame
    diff_frames1 = cv2.absdiff(next_frame, cur_frame)

    # Absolute difference between current frame and # previous frame
    diff_frames2 = cv2.absdiff(cur_frame, prev_frame)

    # Return the result of bitwise 'AND' between the # above two resultant images
    return cv2.bitwise_and(diff_frames1, diff_frames2)

# Capture the frame from webcam
def get_frame(cap):
    # Capture the frame
    ret, frame = cap.read()

    # Resize the image
    #frame = cv2.resize(frame, None, fx=scaling_factor,
     #       fy=scaling_factor, interpolation=cv2.INTER_AREA)

    # Return the grayscale image
    return frame



if __name__=='__main__':
    # Load the model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    model.conf = 0.2  # NMS confidence threshold
    model.max_det = 100  # maximum number of detections per image

    # Define video source inputs
    cap = cv2.VideoCapture(0)
    capwidth, capheight = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap2 = cv2.VideoCapture(2)
    cap2width, cap2height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # We will use temp to swap our camera inputs
    temp = None
    processalgo = 0 # Do we want to use frame differencing or our model?
    switching = False # Are we switching from our model to frame differencing? We need to take 3 more frames of input
    
    # Bounding Box Color
    color = (0,0,255)   

    # Get the first 3 frames 
    prev_frame = get_frame(cap)
    cur_frame = get_frame(cap)
    next_frame = get_frame(cap)

    
    
    # Iterate until the user presses the ESC key
    while True:
        # Display the result of frame differencing
        if processalgo == 0:

        # TODO: 
        # - Create a sampling period
        # - 
        #
            if switching == True: # We resample the next 3 frames to determine whether there is motion
                prev_frame = get_frame(cap)
                cur_frame = get_frame(cap)
                next_frame = get_frame(cap) 
                yoloframecount = 25  # Resets num of frames left to process if switches to yolo
                switching = False
            output_frame = frame_diff(prev_frame, cur_frame, next_frame)

            # sum the grayscale values in our output frame
            b = np.sum(cv2.split(output_frame))
            print(b)

            if b > 800000:        # TODO: Implement a damper system
                print("IT WAS HIT IT WAS HIT")
                processalgo = 1
                b = 36000000

            cv2.imshow("Object Movement", output_frame)

            # Update the variables
            prev_frame = cur_frame
            cur_frame = next_frame
            next_frame = get_frame(cap) 

            

        elif processalgo == 1:
            print("hi")
            output_frame = get_frame(cap)   # Might be skipping a single frame
            prev_out = model(output_frame, size=(1920, 1080)).pandas().xywh[0]
            print(prev_out)
            if not prev_out.empty:
                for _, row in prev_out.iterrows():
                    x_center, y_center, width, height = row['xcenter'], row['ycenter'], row['width'], row['height']

                    # Get top left corner coordinates
                    topLeft = (int(x_center - width/2), int(y_center - height/2))
                    bottomRight = (int(x_center + width/2), int(y_center + height/2))

                    # Draw bounding box
                    cv2.rectangle(output_frame, topLeft, bottomRight, color, 2)

                    # Add class name label
                    cv2.putText(output_frame, str(row['name']), (topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            yoloframecount -= 1
            print(yoloframecount)
            if yoloframecount <= 0:
                processalgo = 0
                switching = True
                
            # Display frame                        
            cv2.imshow("Object Movement", output_frame)

        
        

        # Check if the user pressed ESC
        key = cv2.waitKey(10)
        if key == 27:
            break
        elif key == ord('d'):
            temp = cap
            cap = cap2
            cap2 = temp

            prev_frame = get_frame(cap)
            cur_frame = get_frame(cap)
            next_frame = get_frame(cap)

            # videoSource += 1
            # print(f"next video source: {videoSource}")
            # try:
            #     cap =cv2.VideoCapture(videoSource)
            #     prev_frame = get_frame(cap)
            #     cur_frame = get_frame(cap)
            #     next_frame = get_frame(cap)
            # except:
            #     videoSource -= 1
            #     cap =cv2.VideoCapture(videoSource)
            #     prev_frame = get_frame(cap)
            #     cur_frame = get_frame(cap)
            #     next_frame = get_frame(cap)
            #     print(f"no other video source, back to video source: {videoSource}")
        elif key == ord('a'):
            temp = cap
            cap = cap2
            cap2 = temp

            prev_frame = get_frame(cap)
            cur_frame = get_frame(cap)
            next_frame = get_frame(cap)
            # videoSource -= 1
            # print(f"next video source: {videoSource}")
            # try:
            #     cap =cv2.VideoCapture(videoSource)
            #     prev_frame = get_frame(cap)
            #     cur_frame = get_frame(cap)
            #     next_frame = get_frame(cap)
            # except:
            #     videoSource += 1
            #     cap =cv2.VideoCapture(videoSource)
            #     prev_frame = get_frame(cap)
            #     cur_frame = get_frame(cap)
            #     next_frame = get_frame(cap)
            #     print(f"no other video source, back to video source: {videoSource}")


    cv2.destroyAllWindows()