"""
File for executing tests. Files within each test folder should be placed in
Pi, then run from this file using exec_file().
"""
import os
import sys
import subprocess
import pandas as pd
import numpy as np
import json

sys.path.append('va_pipeline/')
from calculate_accuracy import *
from sensor import *
from test_utils import *

SSH_PI3 = "ssh pi@172.28.69.200"
SSH_PI4 = "ssh pi@172.28.81.58"
ROOT = '~/sysml/testing/'
REPLACE = True


def run_mod(
    res_width: int,
    res_height: int,
    model: str,
    source: str,
    ground_truth: str,
    test_dir: str,
    framerate: int,
    max_frames: int,
    conf: float,
    save_results: bool,
    get_map:bool
    ):
    """Runs mod.py given parameters

    Args:
        res_width (int): input x resolution
        res_height (int): input y resolution
        model (str): desired yolo model
        source (str): sample video (sparse, medium, noisy)
        dest (str): folder destination for inference (relative to config_testing/)
        framerate (int): input framerate
        max_frames (int): max number of frames to process (inclusive)
        save_results (bool): bool to save or discard model outputs
    """
    
    # Output paths of results
    source_name = os.path.splitext(os.path.basename(source))[0]
    test_name = f'temp'
    test_path = os.path.join(test_dir, test_name)

    # Check for existing files
    if not REPLACE:
        matching_files = [filename for filename in os.listdir(test_dir) if filename.startswith(test_name)]
        assert len(matching_files) == 0, 'Test files already in directory.' 

    # Run test
    runtime, energy, out = exec_file(SSH_PI4 + ' ' 
                                    + 'python ./sysml/va_pipeline/mod_pi.py '
                                    + f'--yolov5-model {model} '
                                    + f'--video-source {source} '
                                    + f'--img-width {res_width} '
                                    + f'--img-height {res_height} '
                                    + f'--fps {framerate} ' 
                                    + f'--max-frames {max_frames} '
                                    + f'--conf {conf}'
                                    )
    
    if save_results:
        # Get model outputs
        save_out = subprocess.run('scp pi@172.28.81.58:' 
                    + '~/sysml/testing/test_results/temp.csv' + ' ' 
                    + test_path 
                    + '_inference.csv'
                    )

        if save_out.returncode != 0:
            print(f'scp failed: {out.stderr}')

        # Calculate statistics and save data
        energy.to_csv(test_path + '_energy.csv')
        energy, avg_power = calculate_stats(test_path + '_energy.csv', runtime)

        # Parse output
        parsed_out = parse_mod(out.stdout)
        no_frames = parsed_out['frames']

        # Calculate mAP
        mAP = -1
        if get_map:
            try:
                gt = get_ground_truth_list(res_width, res_height, ground_truth, no_frames)

                # Get preds list
                pred_path = f'{test_path}' + '_inference.csv'
                preds = get_predictions_list(res_width, res_height, pred_path, no_frames)

                # Calculate mAP scores
                mAP = calculate_accuracy(gt, preds)

            except Exception as e:
                print('mAP failed' + str(e))
                pass

        # Write inference times to file        
        with open(test_path + '_stats.txt', 'w') as file:
            file.write(
                    f'frames: {no_frames}\n'
                    + f'frames processed: {str(parsed_out["frames processed"])}'
                    + f'runtime (inference): {str(parsed_out["runtime (inference)"])}\n'
                    + f'average time per frame: {parsed_out["average time per frame"]}\n'  
                    + f'runtime (total): {runtime}\n'
                    + f'energy: {energy}\n' 
                    + f'avg power: {avg_power}\n'
                    + f'energy per frame: {energy / parsed_out["frames"]}\n'
                    + f'mAP: {mAP}'
                    )
            

if __name__ == '__main__':
    dir_to_vid = './sysml/samples/testing/videos/'
    dir_to_gt = './samples/testing/ground_truth/'
    test_dir = './testing/test_results/config_testing/framerate/'
    
    run_mod(
        res_width=640,
        res_height=480,
        model='yolov5n',
        source='./sysml/samples/testing/videos/large_fast.mp4',
        ground_truth='./samples/testing/ground_truth/large_fast.csv',
        test_dir='./',
        framerate=1,
        max_frames=250,
        conf=0.6,
        save_results=True,
        get_map=True
    )