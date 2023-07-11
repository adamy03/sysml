from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip

# Provide the path to the original video file
video_path = "C:/Users/holli/sysml/samples/YouTube/4K Road traffic video for object detection and tracking - free download now! [MNn9qKG2UFI].mp4"
video = VideoFileClip(video_path)
duration = video.duration

print(f"Video duration: {duration} seconds")


# Define the duration of each part in seconds
part_duration = 15
num_clips = duration//part_duration

# Loop through the parts and extract subclips
for i in range(int(num_clips)):
    start_time = i * part_duration
    end_time = (i + 1) * part_duration

    output_path = f"YouTube/Road_traffic_15/segments/part_{i+1}.mp4"  # Output file name for each part

    # Extract the subclip using ffmpeg_extract_subclip function from moviepy
    if i == 12:
        ffmpeg_extract_subclip(video_path, start_time, end_time, targetname=output_path)

    print(f"Part {i+1} extracted successfully.")
