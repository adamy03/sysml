import os
import sys
import subprocess
import pandas as pd
import numpy as np

"""
Helper functions for testing.
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


def parse_mod(out_str):
    split = out_str.split('\n')
    frames = split[0].split(': ')
    runtime = split[1].split(': ')
    avg = split[2].split(': ')
    return {
        frames[0]: int(frames[1]),
        runtime[0]: float(runtime[1]),
        avg[0]: float(avg[1])
    }