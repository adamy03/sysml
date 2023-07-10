import os
import subprocess

# Get all file names from the directory
folder_path = '../samples/YouTube/Road_traffic_15/segments/'
file_names = os.listdir(folder_path)

# Add complete path to the file names if required
#file_names = [os.path.join(folder_path, file_name) for file_name in file_names]

# Initialize an empty list to store the commands
commands = []
#(1280, 720), (960, 540), (640, 360)
#0.3, 0.4, 0.5, 0.6, 0.7

# Loop through all file names
for res in [(1920, 1080)]:
    for conf in [0.6]:
        for file_name in file_names:
            # Form the command and add to the list
            print(file_name)
            file_name = file_name[0:-4]
            command = f"python C:/Users/holli/sysml/va_pipeline/mod.py --img-width {res[0]} --img-height {res[1]} --yolov5-model yolov5x --conf {conf} --video-source \"{file_name}\""
            commands.append(command)

# Run each command using subprocess
for command in commands:
    print(command)
    subprocess.run(command, shell=True)
