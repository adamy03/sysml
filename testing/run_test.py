"""
File for executing tests. Files within each test folder should be placed in pi then run from this
file using exec_file().
"""

from sensor import exec_file

"""
ex:
exec_file("ssh pi@172.28.69.200 'python /home/pi/sysml/image_classification/testing/image_classification_test.py'")
"""

SSH_PI3 = "ssh pi@172.28.69.200"

if __name__ == '__main__':
    exec_file(SSH_PI3 + ' ' + "'python /home/pi/sysml/testing/model_test/inference_test.py'")