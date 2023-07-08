from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip

# Provide the path to the original video file
video_path = "C:/Users/holli/sysml/samples/YouTube/Road traffic Segments/Road traffic video for object recognition [wqctLW0Hb_0].mp4"



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

    output_path = f"YouTube/Road traffic Segments/Segments/part_{i+1}.mp4"  # Output file name for each part

    # Extract the subclip using ffmpeg_extract_subclip function from moviepy
    ffmpeg_extract_subclip(video_path, start_time, end_time, targetname=output_path)

    print(f"Part {i+1} extracted successfully.")
