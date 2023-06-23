import argparse
import os
import sys
from pathlib import Path
from ..testing.model_test.resnet.resnet_images_test import *


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
        yolov5l, 
        resnet50,
        args_str
        ):
    
    function_dict = {

    }
    
    if compress:
        print("compressed")
    else:
        print("not Compressed")

    # Preprocessed image to model
    for input_str in args_str:
        if input_str in function_dict:
            function_dict[input_str]()

    def resnet50(img):
        return model_out(img)

    def yolov5s(img):
        out = exec_file(SSH_PI4 + ' ' + "'python3 detect.py --weights yolov5s.pt --conf 0.2'") #yolo5s



"""
Parses the arguments into variables, for new logic simply add a new arugment
Look through yolov5/detect.py for guidance on adding new arguments
"""
def parse_opt(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)') # --source [directory]
    parser.add_argument('--compress', default=False, action='store_true',  help='True or False') # --compress
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold') # --conf-thres 0.30
    parser.add_argument('--crop', default=False, action='store_true',  help='True or False') # --crop  
    parser.add_argument('--yolov5s', default=True, action='store_true', help='yolov5s False')
    parser.add_argument('--yolov5n', default=False, action='store_true', help='yolov5n False')
    parser.add_argument('--yolov5m', default=False, action='store_true', help='yolov5m False')
    parser.add_argument('--yolov5l', default=False, action='store_true', help='yolov5m False')
    parser.add_argumetn('--resnet50', default=False, action='store_true', help='yolov5m False')
    opt = parser.parse_args()

    args_str = []
    for arg in sys.argv[1:]:
        if arg.startswith('--') and arg.lstrip('--') in vars(opt):
            args_str.append(arg)
    
    print_args(vars(opt))
    return opt, args_str


def main(opt):
    run(**vars(opt), args_str)


if __name__ == '__main__':
    opt, args_str = parse_opt() #Calls the parsing of arguments
    main(opt, args_str)         #Feeds argument variables into run
    
    for arg in args_str:
        print(arg)