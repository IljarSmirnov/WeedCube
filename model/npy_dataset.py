import glob
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import os

class_mapping = {
    "canola": 0,
    "kochia": 1,
    "ragweed": 2,
    "redroot": 3,
    "soybean":4,
    "sugarbeet":5,
    "waterhemp":6,
    "ground":7
}


folders = ["canola","kochia","ragweed","redroot","soybean","sugarbeet","waterhemp","ground"] 
X = []
y = []

for folder in folders:
    csv_files = glob.glob(f"{folder}/*.csv")
    for csv_file in csv_files:
        class_label = class_mapping[f"{folder}"]
       
        df = pd.read_csv(csv_file, sep=',', header=0)
        signatures = df.values
        
        X.extend(signatures)
        y.extend([class_label] * signatures.shape[0])

X = np.array(X)
y = np.array(y)
np.save('X_data.npy', X)
np.save('y_data.npy', y)


print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Пример сигнатуры: {X[0][:5]}...") 
print(f"Пример метки: {y[0]}")