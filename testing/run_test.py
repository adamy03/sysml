"""
File for executing tests. Files within each test folder should be placed in
Pi, then run from this file using exec_file().
"""
import os
import sys
import subprocess
import pandas as pd
import numpy as np

sys.path.append('va_pipeline/')
from calculate_accuracy import *
from sensor import *

SSH_PI3 = "ssh pi@172.28.69.200"
SSH_PI4 = "ssh pi@172.28.81.58"
ROOT = '~/sysml/testing/'
REPLACE = True

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


def run_mod(
    res_width: int,
    res_height: int,
    model: str,
    source: str,
    dest: str,
    framerate: int,
    frame_cap: int,
    conf: float,
    save_results: bool,
    ):
    """Runs mod.py given parameters

    Args:
        res_width (int): input x resolution
        res_height (int): input y resolution
        model (str): desired yolo model
        source (str): sample video (sparse, medium, noisy)
        dest (str): folder destination for inference (relative to config_testing/)
        framerate (int): input framerate
        frame_cap (int): max number of frames to process (inclusive)
        save_results (bool): bool to save or discard model outputs
    """
    
    test_dir = f'./testing/test_results/config_testing/{dest}/'
    test_name = f'diff_{source}_{model}_{res_width}_{res_height}_{framerate}fps'
    test_path = test_dir + test_name

    # Check for existing files
    if not REPLACE:
        matching_files = [filename for filename in os.listdir(test_dir) if filename.startswith(test_name)]
        assert len(matching_files) == 0, 'Test files already in directory.' 

    # Run test
    runtime, energy, out = exec_file(SSH_PI4 + ' ' 
                                    + 'python ./sysml/va_pipeline/mod.py '
                                    + f'--yolov5-model {model} '
                                    + f'--video-source {source} '
                                    + f'--img-width {res_width} '
                                    + f'--img-height {res_height} '  
                                    + f'--frame-cap {frame_cap} '
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
        mAP = 0
        try:
            gt_path = f'./testing/test_results/config_testing/{source}_yolov5l_ground_truth.csv'
            gt = get_ground_truth_list(res_width, res_height, gt_path, no_frames)

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
                    + f'runtime (inference): {str(parsed_out["runtime (inference)"])}\n'
                    + f'average time per frame: {parsed_out["average time per frame"]}\n'  
                    + f'runtime (total): {runtime}\n'
                    + f'energy: {energy}\n' 
                    + f'avg power: {avg_power}\n'
                    + f'energy per frame: {energy / parsed_out["frames"]}\n'
                    + f'mAP: {mAP}'
                    )
    return 

if __name__ == '__main__':
    run_mod(
        res_width=1280,
        res_height=720,
        model='yolov5n',
        source='sparse',
        dest='frame_diff/sparse',
        framerate=25,
        frame_cap=250,
        conf=0.4,
        save_results=True
    )
    clear_chart()
    time.sleep(5)
    run_mod(
        res_width=1280,
        res_height=720,
        model='yolov5n',
        source='medium',
        dest='frame_diff/medium',
        framerate=25,
        frame_cap=250,
        conf=0.4,
        save_results=True
    )
    clear_chart()
    time.sleep(5)
    run_mod(
        res_width=1280,
        res_height=720,
        model='yolov5n',
        source='noisy',
        dest='frame_diff/noisy',
        framerate=25,
        frame_cap=250,
        conf=0.4,
        save_results=True
    )