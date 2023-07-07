import os
import subprocess

# Get all file names from the directory
folder_path = '../samples'
file_names = os.listdir(folder_path)

# Add complete path to the file names if required
#file_names = [os.path.join(folder_path, file_name) for file_name in file_names]

# Initialize an empty list to store the commands
commands = []

# Loop through all file names
for conf in [0.4]:
    for file_name in file_names:
        # Form the command and add to the list
        print(file_name)
        command = f"python C:/Users/shiva/sysml/va_pipeline/mod.py --img-width 1920 --img-height 1080 --yolov5-model yolov5x --conf {conf} --video-source \"{file_name}\""
        commands.append(command)

# Run each command using subprocess
for command in commands:
    print(command)
    subprocess.run(command, shell=True)
