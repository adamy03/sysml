import os
import sys
import argparse
import cv2
import torch
import time
import pandas as pd

from pathlib import Path
from process import *
import subprocess

# Set up path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# Function to run the mod.py command
def run_mod(res, video_source):
    command = f"python C:/Users/holli/sysml/va_pipeline/mod.py --img-width {res[0]} --img-height {res[1]} --video-source '{video_source}'"
    subprocess.run(command, shell=True)

for video in ['sparse', 'medium', 'noisy']:
    for res in [(1280, 720),(960, 540), (640, 360)]:
        run_mod(res, "../../sysml/samples/sparse.mp4") 
