import pandas as pd
import numpy as np

# Run from test_results directory
FILE_PATH = "yolov5n/5sec_1080.csv"

data = pd.read_csv(FILE_PATH)
area = np.trapz(data['Power (W)'], data['Read times - Current graph'])
print("Energy consumption: ", area)
