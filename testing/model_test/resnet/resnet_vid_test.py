import torch
import torchvision
import torchvision.transforms as transforms
import cv2
import time
from PIL import Image

"""
This file opens a video and runs the Resnet model (of varying model sizes) on
each frame for object classification.
Note: Can test different model sizes by changing the Resnet model number
      (near the bottom)
      Can test just preprocessing (for running the model) by commenting
      out this line: score = predict(frame_pil)
The program loops through the video frames. For each frame, it preprocesses the
frame then runs the model on it.
"""


"""
Function to run a single image through model and get top object
classification match.
"""


def predict(transformed_img):
    # print("Running model ---------")
    out = resnet_model(transformed_img)

    softmaxed = torch.nn.functional.softmax(out, dim=1)
    score, index = torch.max(softmaxed, 1)

    sorted_scores, top_indices = torch.sort(softmaxed, descending=True)

    topscore = []
    for i, idx in enumerate(top_indices[0][:1]):
        topscore += [[labels[idx], sorted_scores[0][i].item()]]
    return topscore


"""
Read in a video and loop through its frames using the OpenCV Library.
Runs predict() function on each frame
"""


def process_video(video_path, model):
    frame_scores = {}

    # Set up video capture object
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cv2.VideoCapture.get(
                        cap, int(cv2.CAP_PROP_FRAME_COUNT)
                        ))
    print(f'Frame count: {frame_count}')

    index = 1
    start_time = time.time()
    
    print("Looping through frames...")

    # Loop through frames
    while cap.isOpened():
        # PREPROCESSING: Read frame data and convert to img, then transform
        ret, frame = cap.read()
        if not ret:  # no more frames to read
            break
        frame_pil = Image.fromarray(frame)
        transformed_img = my_transform(frame_pil).unsqueeze(0)

        # Keep track of what frame number we are on
        if (index % 10 == 0):
            print("Frame number: ", index)
        index += 1

        # Run prediction on this frame
        # COMMENT OUT the call to predict() if only testing preprocessing
        score = None
        score = predict(transformed_img)
        frame_scores[index] = score

    total_time = time.time() - start_time
    print(f'Total time: {total_time}')
    print(f'Time per frame: {total_time / frame_count}')

    cap.release()
    return [frame_count, frame_scores]


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
resnet_model = torchvision.models.resnet18(pretrained=True)
resnet_model.eval()


"""
Run the video through the model and get results
"""
frame_scores = process_video('/home/pi/5sec_vid.mp4', resnet_model)
# print(frame_scores)
