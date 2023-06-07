"""
This file tests energy usage of Raspberry Pi while it
runs the ResNet50 model (pretrained on ImageNet) to
perform object detection on a presaved image.
"""

from PIL import Image
import torch
import torchvision
import pandas as pd
import time


"""
Prepares the program for inference: imports model, opens labels,
and creates the transform.
"""

def inference_setup():
    # Import resnet50 model
    resnet50_model = torchvision.models.resnet50(pretrained=True)
    resnet50_model.eval()

    # Open labels
    with open('/home/pi/sysml/image_classification/testing/labels.txt', 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Create the transform
    my_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


"""
Function that runs the ResNet50 model on one image.
"""
def model_out(img):
    transformed_img = my_transform(img).unsqueeze(0)
    out = resnet50_model(transformed_img)


"""
Runs the model on an image 'num_imgs' number of times and 
saves the results.
"""
def test_inference(num_imgs):
    # Run inference on an image 50 times
    out = []
    for i in range(5):
        # open image
        img = Image.open("/home/pi/sysml/testing/model_test/golden.jpeg")
        # append inference to out array
        out.append(model_out(img)[0])

    # Convert out array to data frame, then save as csv
    df = pd.DataFrame(out)
    df.to_csv('/home/pi/sysml/testing/testing/model_test/model_output.csv')


"""
Define execution of desired tests here
"""
def run_tests():
    test_inference(5)


if __name__ == '__main__':
    run_tests()
