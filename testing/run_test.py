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

    # out = exec_file(SSH_PI3 + ' ' + "'python ~/sysml/testing/sensing/video_test/taking_video_test.py'")
    # out = exec_file(SSH_PI3 + ' ' + "'python ~/sysml/testing/model_test/ssdvgg_vid_res_test.py'")
    
    out = exec_file(SSH_PI3 + ' ' + "'python ~/sysml/testing/model_test/yolov5-on-rpi4-2020/johnston_yolov5/yolov5/detect.py --weights yolov5n.pt --conf 0.2 --source /home/pi/sysml/testing/sensing/video_test/5sec_480.mp4")
    

    # Change file name/path as needed, assumes run_test is run from sysml directory
    out.to_csv("~/sysml/testing/test_results/yolov5n/5sec_480.csv")
    #out.to_csv("~/sysml/testing/test_results/yolov5n/5sec_720.csv")
    #out.to_csv("~/sysml/testing/test_results/yolov5n/5sec_1080.csv")
