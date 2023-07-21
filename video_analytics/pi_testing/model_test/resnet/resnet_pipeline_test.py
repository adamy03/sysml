import torch
import torchvision
import torchvision.transforms as transforms
import cv2
import time
from PIL import Image

"""
This file opens a video and runs the Resnet model (of varying model sizes) on
each frame for object classification.

Note: The program separates the preprocessing and inference stages by first
preprocessing all of the frames in the video, then running the model on all
frames.

"""


"""
Conducts preprocessing for all frames in the video.
"""

def compress_image(frame, scale, w, h):
    resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
    return resized


def preprocess_frames(cap):  
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frames = []
    index = 1
    print("Looping through frames...")

    # Loop through frames
    while cap.isOpened():
        # Read frame data and convert to img, then transform
        ret, frame = cap.read()
        if not ret:  # no more frames to read
            break

        compress_image(frame, 0.5, frame_width, frame_height)
        frame_pil = Image.fromarray(frame)
        transformed_img = my_transform(frame_pil).unsqueeze(0)
        frames.append(transformed_img)

        # Keep track of what frame number we are on
        if (index % 10 == 0):
            print("Frame number: ", index)
        index += 1
    return frames


"""
Runs model on all frames in the video
"""


def predict(frames):
    topscores = []
    index = 1
    for f in frames:
        out = resnet_model(f)

        softmaxed = torch.nn.functional.softmax(out, dim=1)
        score, ind = torch.max(softmaxed, 1)

        sorted_scores, top_indices = torch.sort(softmaxed, descending=True)

        for i, idx in enumerate(top_indices[0][:1]):
            topscores += [[labels[idx], sorted_scores[0][i].item()]]
            
        # Keep track of what frame number we are on
        if (index % 10 == 0):
            print("Frame number: ", index)
        index += 1
    return topscores


"""
Function that runs the video analysis; preprocesses all of the frames
and then runs the model on each frame
"""


def process_video(video_path, model):
    frame_scores = {}
    
    # Set up video capture object
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cv2.VideoCapture.get(
                        cap, int(cv2.CAP_PROP_FRAME_COUNT)
                        ))
    print(f'Frame count: {frame_count}')
    
    # Preprocess all of the frames
    print("Looping through frames for preprocessing")
    start_time = time.time()
    frames = preprocess_frames(cap)
    total_time = time.time() - start_time
    print(f'Total time for preprocessing: {total_time}')
    print(f'Time per frame: {total_time / frame_count}')
    
    # Wait
    print("Pausing...")
    time.sleep(3)

    # Run prediction on all of the frames
    print("Looping through frames for prediction")
    start_time = time.time()
    topscores = predict(frames)
    total_time = time.time() - start_time
    print(f'Total time for prediction: {total_time}')
    print(f'Time per frame: {total_time / frame_count}')

    cap.release()
    return topscores


"""
Setup for running the model
"""

print("Setting up the transform...")

# Define the torchvision image transforms
transform = transforms.Compose([transforms.ToTensor(), ])

# Create the transform
my_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
])

print("Opening labels.txt...")

# Open labels
with open('/home/pi/sysml/testing/model_test/labels.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]


"""
Set up the Resnet model.
FOR TESTING DIFF MODEL SIZES: update the model number below
ex: Resnet 18, 34, 50, 101, 152 (# of layers in each)
"""

print("Setting up Resnet model...")

# Load the object classification model and set mode to eval
resnet_model = torchvision.models.resnet50(pretrained=True)
resnet_model.eval()


"""
Run the video through the model and get results
"""
topscores = process_video('/home/pi/5sec_vid.mp4', resnet_model)
# print(frame_scores)
