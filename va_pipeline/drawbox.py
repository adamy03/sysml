import cv2
import pandas as pd

def draw_boxes(video_path, df_box1, df_box2, out_path):
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    resX, resY, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and output video file
    output_video = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (resX, resY))

    frameCount = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or cv2.waitKey(1) & 0xFF == ord('q'): break

        # Selected colors
        blue = (255, 255, 0)
        green = (0, 255, 0)

        for df, color in [(df_box1, blue), (df_box2, green)]:  # Draws bounding boxes from both models each frame
            currRow = df[df['frame'] == frameCount] # selects all bounding boxes as rows for a given frame
            for _, row in currRow.iterrows():
                x_center, y_center, width, height = row['xcenter'], row['ycenter'], row['width'], row['height']

                # Draws individual bounding box
                cv2.rectangle(frame, (int(x_center - width/2), int(y_center- height/2)), (int(x_center + width/2), int(y_center + height/2)), color, 2)

        output_video.write(frame)
        cv2.imshow('Current Frame', frame) 
        frameCount += 1

    cap.release()
    output_video.release()
    cv2.destroyAllWindows()


# Data Selection/ Output

video_path = '../samples/DE_sample1.mp4'
df_boxes1 = pd.read_csv('../testing/test_results/yolov5x_DE.csv')   # Blue Bounding Box CSV Path
df_boxes2 = pd.read_csv('../testing/test_results/yolov5s_DE.csv')   # Green Bounding Box CSV Path
output_path = '../samples/output_video.mp4'


# Run the file
draw_boxes(video_path, df_boxes1, df_boxes2, output_path)
