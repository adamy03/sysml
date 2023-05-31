import io
import logging
import socketserver
from threading import Condition
from http import server
import time

import cv2
from PIL import Image
import torch
import torchvision

resnet50_model = torchvision.models.resnet50(pretrained=True)
resnet50_model.eval()

with open('labels.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Create the transform
my_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def modelOut(frame):
    img = Image.fromarray(frame)
    transformed_img = my_transform(img).unsqueeze(0)
    out = resnet50_model(transformed_img)

    softmaxed = torch.nn.functional.softmax(out, dim=1)
    score, index = torch.max(softmaxed, 1)

    sorted_scores, top_indices = torch.sort(softmaxed, descending=True)

    outList = []
    for i, idx in enumerate(top_indices[0][:5]):
        outList += [[labels[idx], sorted_scores[0][i].item()]]

    return outList

PAGE = """\
<html>
<head>
<title>Raspberry Pi - Pi Camera</title>
<script>
function updateOutput(content) {
  document.getElementById("output").innerHTML = content;
}
</script>
</head>
<body>
<center><h1>Raspberry Pi - Pi Camera</h1></center>
<center><img src="stream.mjpg" width="640" height="480"></center>
<div id="output"></div>
</body>
</html>
"""

class StreamingOutput:
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, frame):
        with self.condition:
            self.frame = frame
            self.condition.notify_all()


class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()

            cap = cv2.VideoCapture(0)  # Initialize the camera capture

            try:
                while True:
                    ret, frame = cap.read()  # Read a frame from the camera

                    if not ret:
                        break

                    # Process the frame and get the model output
                    mOut = modelOut(frame)

                    # Update the output on the HTML page
                    content = ', '.join(f'{label}: {score:.2f}' for label, score in mOut)
                    output_script = f'<script>updateOutput("{content}")</script>\r\n'
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(output_script.encode('utf-8')) + len(frame))
                    self.end_headers()
                    self.wfile.write(output_script.encode('utf-8'))
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')

            except Exception as e:
                logging.warning('Removed streaming client %s: %s', self.client_address, str(e))

            finally:
                cap.release()

        else:
            self.send_error(404)
            self.end_headers()

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

with StreamingServer(('0.0.0.0', 8000), StreamingHandler) as server:
    server.serve_forever()
