import pyautogui
import subprocess
import win32gui
import time
import pandas as pd
import multiprocessing

SSH_PI3 = "ssh pi@172.28.69.200"
SSH_PI4 = "ssh pi@172.28.81.58"

# Set the title of the window you want to click within
UM25_WINDOW = 'UM25C PC Software V1.3'

CONNECT_X = 225
CONNECT_Y = 75
GRAPH_X = 500
GRAPH_Y = 75

# Example for running image classification in pi3
# command = "ssh pi@172.28.69.200 'python /home/pi/sysml/ModelClassification/testModel/image_classification_test.py'"

# To start/stop Temp Sensing
is_running = True

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
    
    # Marks the end of the program running
    global is_running
    is_running = False 


"""
Returns graph of voltage, current, and power from UM25 software.
TODO: Implement proper scaling of readTimes to Sec, align Temp_data, align "Event" data
"""
def get_energy(exec_time: int = 0, scale_axis: bool=False):
    window_handle = win32gui.FindWindow(None, UM25_WINDOW)
    
    window_rect = win32gui.GetWindowRect(window_handle)
    window_x = window_rect[0]
    window_y = window_rect[1]

    # Calculate the absolute coordinates inside the window
    x = window_x + GRAPH_X
    y = window_y + GRAPH_Y

    # Move the mouse to the specified coordinates and click
    pyautogui.moveTo(x, y)
    pyautogui.click()
    pyautogui.rightClick()

    # Move to copy
    x = x + 20
    y = y + 30
    pyautogui.moveTo(x, y)

    # Copy to clipboard
    pyautogui.click()

    df = pd.read_clipboard()

    df.drop(labels='Unnamed: 4', axis=1, inplace=True)
    df['Power (W)'] = df['Voltage(V) - Voltage  graph'] * df['Current(A) - Current graph'] 

    if scale_axis:
        factor = df['Read times - Voltage  graph'][-1] / exec_time
        df['Read times - Voltage  graph'], df['Read times - Current  graph'] = factor * df['Read times - Voltage  graph']

    return df

"""
Returns a Dataframe Column of CPU Temp Readings
"""
def check_cpu_temp(command):
    command = command[:-3] #removes .py to command 
    command += ".txt" 
    tempList = []
    while is_running:
        temp = subprocess.run(command)
        tempList += [temp]
        time.sleep(1)
    df2 = pd.DataFrame(tempList)
    return df2

"""
Runs file on Pi from local machine, and begins UM25 logger. Note that logger
software must be open in separate window and within view (eg: split the screen
between the terminal and logger so the click will register on connect)
#TODO add option for logging temp
"""
def exec_file(command):
    start = time.time()
    # temperature_process = multiprocessing.Process(target=check_cpu_temp)

    # Start the temperature checking process
    # temperature_process.start()

    print('starting sensor...')
    click_button_start(UM25_WINDOW, CONNECT_X, CONNECT_Y)

    print('running file...')
    subprocess.run(command, shell=True)

    print('stopping sensor')
    click_button_end(UM25_WINDOW, CONNECT_X, CONNECT_Y)
    end = time.time()

    #TODO sort out this section
    # Wait for the temperature checking process to finish
    # temperature_process.join()

    # Returns a Column of temps every second from the temperature checking process
    # cpu_temp_data = temperature_process.get()
    # Appends cpu_temp column
    # cpu_temp_data = pd.Series(cpu_temp_data) #aligns data based on index, not proper time
    # df['Temperature/Sec'] = cpu_temp_data

    return get_energy()


