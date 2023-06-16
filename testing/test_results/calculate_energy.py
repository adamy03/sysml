import pandas as pd
import numpy as np

# Make sure to run from sysml directory
FILE_PATH = "/testing/test_results.csv"

data = pd.read_csv(FILE_PATH)
area = np.trapz(data['y'], data['x'])
print("Energy consumption: ", area)
