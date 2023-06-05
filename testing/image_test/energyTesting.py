import pyautogui
import subprocess

def click_button(x, y):
    # Move the mouse to the specified coordinates and click
    pyautogui.moveTo(x, y)
    pyautogui.click()

# Set the coordinates where the button is located on your screen
button_x = 100
button_y = 200

# Click the button to start the testing process
click_button(button_x, button_y)

# Specify the path to your Python file
python_file_path = "testModel.py"

# Run the Python file using subprocess
subprocess.call(["python", python_file_path])
