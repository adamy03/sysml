import pyautogui
import subprocess
import win32gui

def click_button(window_title, relative_x, relative_y):
    # Find the window by its title
    window_handle = win32gui.FindWindow(None, window_title)

    if window_handle == 0:
        print(f"Window '{window_title}' not found.")
        return

    # Get the window's position
    window_rect = win32gui.GetWindowRect(window_handle)
    window_x = window_rect[0]
    window_y = window_rect[1]

    # Calculate the absolute coordinates inside the window
    x = window_x + relative_x
    y = window_y + relative_y

    # Move the mouse to the specified coordinates and click
    pyautogui.moveTo(x, y)
    pyautogui.click()
    pyautogui.click()

# Set the title of the window you want to click within
window_title = "UM25C PC Software V1.3"

# Set the relative coordinates within the window
relative_x = 225
relative_y = 75

# Click the button inside the window to start the testing process
click_button(window_title, relative_x, relative_y)

# Specify the path to your Python file
python_file_path = "testModel.py"

# Run the Python file using subprocess
subprocess.call(["python", python_file_path])
