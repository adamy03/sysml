import cv2
import pandas as pd
import time
import pdb

videoPath = '../samples/DE_sample1.mp4'
dfBoxes1 = pd.read_csv('../testing/test_results/yolov5x_DE.csv')
dfBoxes2 = pd.read_csv('../testing/test_results/yolov5s_DE.csv')

cap = cv2.VideoCapture(videoPath)

# Get video properties
resX = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
resY = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(resX, resY, fps)

# Define the codec and output video file
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter('../samples/output_video.mp4', fourcc, fps, (resX, resY))


#print(dfBoxes1)

frameCount = 1
while cap.isOpened():
    #print(frameCount)
    ret, frame = cap.read()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    if not ret:
        # If there are no frames you break the loop
        break

    print(frameCount)
    
    #pdb.set_trace()
    currRow1 = dfBoxes1[dfBoxes1['frame'] == frameCount]
    # x_center = currRow1.iloc[1]['xcenter']
    # print(f'This is the x_center: {x_center}')
    # print(currRow1)
    # print(currRow1.shape[0])

    for index in range(currRow1.shape[0]):
        print(index)
        x_center = currRow1.iloc[index]['xcenter']
        y_center = currRow1.iloc[index]['ycenter']
        width = currRow1.iloc[index]['width']
        height = currRow1.iloc[index]['height']
        #print(x_center, y_center, width, height)

        cv2.rectangle(frame, (int(x_center - width/2), int(y_center- height/2)), (int(x_center + width/2), int(y_center + height/2)), (255, 255, 0), 2)


    # for index, row in currRow1.iteritems():
    #     x_center = currRow1.loc[row, 'xcenter']
    #     print(x_center)
    #     y_center = currRow1.loc[row, 'ycenter']
    #     print(y_center)
    #     width = currRow1.loc[row, 'width']
    #     print(width)
    #     height = currRow1.loc[row, 'height']
    #     print(height)

    #     cv2.rectangle(frame, (int(x_center), int(y_center)), (int(x_center + width), int(y_center + height)), (255, 255, 0), 2)

    # Verify if the box is within the frame
    
    #pdb.set_trace()
    currRow2 = dfBoxes2[dfBoxes2['frame'] == frameCount]
    #print(currRow2)
    #print(frameCount)

    for index in range(currRow2.shape[0]):
        x_center = currRow2.iloc[index]['xcenter']
        y_center = currRow2.iloc[index]['ycenter']
        width = currRow2.iloc[index]['width']
        height = currRow2.iloc[index]['height']

        cv2.rectangle(frame, (int(x_center), int(y_center)), (int(x_center + width), int(y_center + height)), (0, 255, 0), 2)

    output_video.write(frame)

    cv2.imshow('Current Frame', frame) 
    
    #time.sleep(1)

    frameCount += 1


cap.release()
output_video.release()
cv2.destroyAllWindows()