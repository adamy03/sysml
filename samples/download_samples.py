"""Downloads videos from youtube link specified in download_links file
"""
import subprocess

directory = "."
file_path = "download_links.txt"  

with open(file_path, "r") as file:
    videoList = [line.rstrip('\n') for line in file]

videoList = videoList[2:] # Filters out videoLinks.txt comments

# Display the list of lines
print(videoList)


for link in videoList:
    subprocess.run(["yt-dlp", link, "--path", directory]) # Download in specific directory
