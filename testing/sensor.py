import pyautogui
import subprocess
import win32gui
import time


# Set the title of the window you want to click within
WINDOW_TITLE = "UM25C PC Software V1.3"
RELATIVE_X = 225
RELATIVE_Y = 75

# Example for running image classification in pi3
# command = "ssh pi@172.28.69.200 'python /home/pi/sysml/ModelClassification/testModel/image_classification_test.py'"

"""
Finds region and clicks in window
"""


def click_button_start(window_title, relative_x, relative_y):
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


def click_button_end(window_title, relative_x, relative_y):
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


"""
Runs file on Pi from local machine, and begins UM25 logger. Note that logger
software must be open in separate window and within view (eg: split the screen
between the terminal and logger so the click will register on connect)
"""
def exec_file(command):
    start = time.time()
    print('starting sensor...')
    click_button_start(WINDOW_TITLE, RELATIVE_X, RELATIVE_Y)

    print('running file...')
    subprocess.run(command, shell=True)

    print('stopping sensor')
    click_button_end(WINDOW_TITLE, RELATIVE_X, RELATIVE_Y)
    print("exec file time: {}".format(time.time() - start))


