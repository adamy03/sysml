from new_pipeline import *
from pipeline_utils import *
import cv2

vid = Video(path='C:/Users/shiva/sysml/samples/videos/large_fast.mp4')
cache = FrameCache(max_frames=5)
model = Model('yolov5n')

# initialize cache
cache = vid.read_frame()
print(len(cache.frames))
print('---------------------')
for c in cache.frames:
    print(c[0])

while vid.ret:
    print(model.run(cache.get_frame(0), vid))
    vid.add_to_cache(frames_skip=5, frame_cache=cache)
    
add_to_cache(5, cache)
