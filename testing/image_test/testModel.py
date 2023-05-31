import cv2
from PIL import Image
import torch
import torchvision

import time
from picamera2 import Picamera2, Preview

picam2 = Picamera2()

preview_config = picam2.create_preview_configuration(main={"size": (800, 600)})



resnet50_model = torchvision.models.resnet50(pretrained=True)
resnet50_model.eval()

with open('labels.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]
print("1")
# Create the transform
my_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("2")

def modelOut(img):
    #img = Image.fromarray(frame)
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

picam2.start()
metadata = picam2.capture_file("test.jpeg")
img = Image.open("test.jpeg")
modelOut(img)

