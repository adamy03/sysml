from new_pipeline import *
from pipeline_utils import *
import cv2

vid = Video(path='~/sysml/samples/videos/large_fast.mp4')
cache = FrameCache(max_frames=5)
model = Model('yolov5n')

# initialize cache
cache = vid.fill_cache(frames_skip=50, frame_cache=cache)
print(len(cache.frames))
print('---------------------')
for c in cache.frames:
    print(c[0])
print('######################')

while vid.ret:
    frame_num, frame = cache.get_frame()
    print(frame_num)
    det = model.run(frame, vid)
    vid.add_to_cache(frames_skip=50, frame_cache=cache)
    
print(model.outputs)
