"""
File for executing tests. Files within each test folder should be placed in
Pi, then run from this file using exec_file().
"""
import os
import subprocess
import pandas as pd
import numpy as np

from sensor import exec_file

SSH_PI3 = "ssh pi@172.28.69.200"
SSH_PI4 = "ssh pi@172.28.81.58"

"""
Just runs python file, nothing else
"""

def run_file(command):
    out = subprocess.run(command, 
                         shell=True, 
                         capture_output = True, 
                         text = True
                         )
    return out


def calculate_stats(fpath, runtime):
    # Energy consumption
    data = pd.read_csv(fpath)
    area = np.trapz(data['Power (W)'], data['Time (s)'])

    return area, area / runtime
  

if __name__ == '__main__':
    # Change to name and path of output files
    TEST_DIR = './testing/test_results/config_testing/noisy/'
    TEST_NAME = 'noisy_yolov5n_1280_720_25fps'
    TEST_PATH = TEST_DIR + TEST_NAME

    # Check for existing files:
    matching_files = [filename for filename in os.listdir(TEST_DIR) if filename.startswith(TEST_NAME)]
    print(matching_files)
    assert len(matching_files) == 0, 'Test files already in directory.' 

    # Run test
    runtime, energy, out = exec_file(SSH_PI4 + ' ' + "'python ~/sysml/va_pipeline/run.py")
    subprocess.run('scp pi@172.28.81.58:' +
                   '~/sysml/testing/test_results/temp.csv' + ' ' +
                   TEST_PATH + 
                   '_inference.csv'
                   ) #get model outputs

    # Calculate statistics and save data
    energy.to_csv(TEST_PATH + '_energy.csv')
    energy, avg_power = calculate_stats(TEST_PATH + '_energy.csv', runtime)

    with open(TEST_PATH + '.txt', 'w') as file:
        file.write(out.stdout + 
                   f'runtime (total): {runtime}\n'
                   f'energy: {energy}\n' +
                   f'avg power: {avg_power}'
                   )
