import torchvision
import torchvision.transforms as transforms
import cv2
import time

"""
This file opens a video and runs the SSD300 VGG16 model on each frame to draw
and label bounding boxes for object detection.
"""


"""
Function to run a single image through model and get boxes, labels, and scores
"""


def predict(image, model, detection_threshold):
    # transform the image to tensor
    image = transform(image)

    # add a batch dimension
    image = image.unsqueeze(0)

    # get the predictions on the image
    outputs = model(image)

    # get score for all the predicted objects
    pred_scores = outputs[0]['scores']
    pred_scores = pred_scores.detach()

    # get all the predicted bounding boxes and filter by threshold
    pred_bboxes = outputs[0]['boxes']
    pred_bboxes = pred_bboxes.detach()
    boxes = pred_bboxes[pred_scores >= detection_threshold]

    # get all predicted labels and filter by threshold
    labels = outputs[0]['labels']
    labels = labels[pred_scores >= detection_threshold]

    scores = pred_scores[pred_scores >= detection_threshold]

    return boxes, labels, scores


"""
Read in a video and loop through its frames using the OpenCV Library.
Runs predict() function on each frame
"""


def should_process_frame(index):
    return (index % 10 == 0)


def process_video(video_path, model):
    results = {}

    # Set up video capture object
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cv2.VideoCapture.get(cap, int(
                        cv2.CAP_PROP_FRAME_COUNT)))
    print(f'Frame count: {frame_count}')

    index = 0
    start_time = time.time()

    # Loop through frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        index += 1
        
        if should_process_frame(index):
            if index % 10 == 0:
                print(f'Reached index {index}')
            results[index] = {}

            # Run prediction on this frame and store results
            boxes, labels, scores = predict(frame, model, 0.3)
            if index % 10 == 0:
                print(f'Boxes: {boxes}\n Labels: {labels}\n Scores: {scores}\n')
            results[index]['boxes'] = boxes
            results[index]['labels'] = labels      
            results[index]['scores'] = scores 

    end_time = time.time()
    print(f'Total time: {end_time - start_time}')

    cap.release()

    return results


"""
Set up the model
"""

# Define the torchvision image transforms
transform = transforms.Compose([transforms.ToTensor(), ])

# Load the object detection model, SSD, and set mode to eval
model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
model.eval()

"""
Run the video through the model and get results
"""
results = process_video('/home/pi/sysml/testing/sensing/video_test/5sec_480.mp4', model)
#results = process_video('/home/pi/sysml/testing/sensing/video_test/5sec_720.mp4', model)
#results = process_video('/home/pi/sysml/testing/sensing/video_test/5sec_1080.mp4', model)
