"""
This file tests energy usage of Raspberry Pi while it
runs the ResNet50 model to perform object detection
on a presaved image.
"""

from PIL import Image
import torch
import torchvision
import pandas as pd
import time


time.sleep(3)

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

# Open labels
with open('/home/pi/sysml/testing/model_test/labels.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Defines model inference pipeline
def modelOut(img):
    transformed_img = my_transform(img).unsqueeze(0)
    out = resnet50_model(transformed_img)

    softmaxed = torch.nn.functional.softmax(out, dim=1)
    score, index = torch.max(softmaxed, 1)

    sorted_scores, top_indices = torch.sort(softmaxed, descending=True)

    outList = []
    for i, idx in enumerate(top_indices[0][:5]):
        outList += [[labels[idx], sorted_scores[0][i].item()]]
    print(outList)
    return outList


# Run inference on an image 50 times
out = []
for i in range(5):
    # open image
    img = Image.open("/home/pi/sysml/image_classification/testing/golden.jpeg")
    # append inference to out array
    out.append(modelOut(img)[0])

# Convert out array to data frame, then save as csv
df = pd.DataFrame(out)
df.to_csv('/home/pi/sysml/testing/model_test/model_output.csv')


"""
Define execution of desired tests here
"""
def run_tests():
    test_camera_image(600, 800, 10)


if __name__ == '__main__':
    run_tests()
    
time.sleep(3)
