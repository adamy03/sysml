"""
File for executing tests. Files within each test folder should be placed in pi then run from this
file using exec_file().
"""

from sensor import exec_file
import subprocess
import time

SSH_PI3 = "ssh pi@172.28.69.200"

"""
Just runs python file, nothing else
"""
def run_file(command):
    subprocess.run(command, shell = True)


if __name__ == '__main__':
    exec_file(SSH_PI3 + ' ' + "'python /home/pi/sysml/testing/sensing/image_test/image_tests.py'")
