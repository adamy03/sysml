import pandas as pd
import numpy as np

# Run from test_results directory
FILE_PATH = "german/german_yolov5n.csv"

data = pd.read_csv(FILE_PATH)
area = np.trapz(data['Power (W)'], data['Time (s)'])
print("Energy consumption: ", area)
