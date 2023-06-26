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
    runtime, energy, out = exec_file(SSH_PI4 + ' ' + "'python ~/sysml/va_pipeline/run.py")
    energy.to_csv('testing/test_results/temp1.csv')
    subprocess.run('scp pi@172.28.81.58:~/sysml/testing/test_results/temp_df.csv ./testing/test_results/temp.csv')
    # Change csv's file name/path as needed, assumes run_test is run from sysml directory
    # filename = "~/sysml/testing/test_results/german/german_yolov5n.csv"
    
    # Calculate statistics
    power, avg_power = calculate_stats('testing/test_results/temp1.csv', runtime)
    with open('testing/test_results/temp.txt', 'w') as file:
        file.write(out.stdout + 
                   f'runtime (total): {runtime}\n'
                   f'power: {power}\n' +
                   f'avg power: {avg_power}'
                   )
        

    
