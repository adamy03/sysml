import argparse
import os
import sys
from pathlib import Path


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

#from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

"""
This is where we execute our code, write all of the function logic/pipeline connection code here!
"""
def run(compress, 
        source,  
        conf_thres, 
        crop,
        yolov5s,
        yolov5n,
        yolov5m,
        yolov5l
        ):
    
    if compress:
        print(True)
    else:
        print(False)

"""
Parses the arguments into variables, for new logic simply add a new arugment
Look through yolov5/detect.py for guidance on adding new arguments
"""
def parse_opt(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--compress', default=False, action='store_true',  help='True or False') # --compress
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)') # --source [directory]
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold') # --conf-thres 0.30
    parser.add_argument('--crop', default=False, action='store_true',  help='True or False') # --crop  
    parser.add_argument('--yolov5s', default=True, action='store_true', help='yolov5s False')
    parser.add_argument('--yolov5n', default=False, action='store_true', help='yolov5n False')
    parser.add_argument('--yolov5m', default=False, action='store_true', help='yolov5m False')
    parser.add_argument('--yolov5l', default=False, action='store_true', help='yolov5m False')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt



def main(opt):
    run(**vars(opt))

if __name__ == '__main__':
    opt = parse_opt() #Calls the parsing of arguments
    main(opt)         #Feeds argument variables into run