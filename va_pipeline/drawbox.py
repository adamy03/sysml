import cv2
import pandas as pd

def draw_boxes(video_path, ground_box, inference_box, out_path):
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    resX, resY, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and output video file
    output_video = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (resX, resY))

    # Selected colors
    blue = (255, 0, 0)
    green = (0, 255, 0)
    
    dataframes = [(ground_box, green)]
    if inference_box is not None:
        dataframes.append((inference_box, blue))

    frameCount = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or cv2.waitKey(1) & 0xFF == ord('q'): break

        for df, color in dataframes:
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

# video_path = '../samples/YouTube/Road_traffic_15/segments/medium_large_slow.mp4'
# ground = pd.read_csv('../testing/test_results/new_video_results/medium_large_slow_yolov5x_1920_1080_0.6conf.csv')   # Green Bounding Box CSV Path
# # inference = pd.read_csv('../testing/test_results/config_testing/resolution/sparse/sparse_yolov5n_960_540_25fps_inference.csv')   # Blue Bounding Box CSV Path

# output_path = '../samples/output_video.mp4'


# # Run the function
# # draw_boxes(video_path=video_path, ground_box=ground, inference_box=inference, out_path=output_path)
# draw_boxes(video_path=video_path, ground_box=ground, inference_box=None, out_path=output_path)
