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
    exec_file(SSH_PI4 + ' ' + "'python3 /home/pi/yolov5-on-rpi4-2020/johnston_yolov5/yolov5/detect.py --weights yolov5s.pt --conf 0.2 --source /home/pi/yolov5-on-rpi4-2020/johnston_yolov5/yolov5/images/shortVidIntersection.mp4'")
