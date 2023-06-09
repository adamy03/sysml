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
    out = exec_file(SSH_PI3 + ' ' + "'python /home/pi/sysml/testing/model_test/resnet_vid_test.py'")

    # Change file name/path as needed, assumes run_test is run from sysml directory
    out.to_csv("./testing/test_results/resnet18_720vid.csv")
