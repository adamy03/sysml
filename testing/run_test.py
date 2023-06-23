"""
File for executing tests. Files within each test folder should be placed in
Pi, then run from this file using exec_file().
"""
import pandas as pd
import numpy as np
from sensor import exec_file
import subprocess

SSH_PI3 = "ssh pi@172.28.69.200"
SSH_PI4 = "ssh pi@172.28.81.58"

"""
Just runs python file, nothing else
"""


def run_file(command):
    subprocess.run(command, shell=True)


def calculate_stats(fpath, runtime):
    # Energy consumption
    data = pd.read_csv(fpath)
    area = np.trapz(data['Power (W)'], data['Time (s)'])
    print("Energy consumption: ", area, " J")
    
    # Runtime
    print("Runtime: ", runtime, " sec")
    
    # Average power
    print("Avg power: ", area / runtime, " W")
  

if __name__ == '__main__':

    # out = exec_file(SSH_PI3 + ' ' + "'python ~/sysml/testing/sensing/video_test/taking_video_test.py'")
    # out = exec_file(SSH_PI3 + ' ' + "'python ~/sysml/testing/model_test/ssdvgg_vid_res_test.py'")
    
    runtime, out = exec_file(SSH_PI4 + ' ' + "'python ~/sysml/testing/model_test/yolov5-on-rpi4-2020/johnston_yolov5/yolov5/detect.py --weights yolov5n.pt --save-txt --save-conf --conf 0.2 --name german_yolov5n --source /home/pi/german.mp4")
    

    # Change csv's file name/path as needed, assumes run_test is run from sysml directory
    filename = "~/sysml/testing/test_results/german/german_yolov5n.csv"
    out.to_csv(filename)
    
    # Calculate statistics
    calculate_stats(filename, runtime)
