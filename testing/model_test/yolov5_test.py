"""
File for executing tests. Files within each test folder should be placed in
Pi, then run from this file using exec_file().
"""

from sensor import exec_file
import subprocess

SSH_PI3 = "ssh pi@172.28.69.200"
SSH_PI4 = "ssh pi@172.28.81.58"

"""
Just runs python file, nothing else
"""


def run_file(command):
    subprocess.run(command, shell=True)

if __name__ == '__main__':

    # These will run model on all images in the 'yolov5/data/images' folder
    
    #out = exec_file(SSH_PI4 + ' ' + "'python3 detect.py --weights yolov5n.pt --conf 0.2'") #yolo5n
    out = exec_file(SSH_PI4 + ' ' + "'python3 detect.py --weights yolov5s.pt --conf 0.2'") #yolo5s
    #out = exec_file(SSH_PI4 + ' ' + "'python3 detect.py --weights yolov5m.pt --conf 0.2'") #yolo5m
    #out = exec_file(SSH_PI4 + ' ' + "'python3 detect.py --weights yolov5l.pt --conf 0.2'") #yolo5l

    # Change file name/path as needed, assumes run_test is run from sysml directory
    out.to_csv("./testing/test_results/auto_test.csv")