# Import Libraries
import os
import pandas as pd
import numpy as np

# Get Dataset from GitHub
Dataset_url = 'https://raw.githubusercontent.com/Akib3n/USA-EUR-Daily-Foreign-Exchange-since-1999/main/daily.csv'

# Load the dataset
Dataset = pd.read_csv(Dataset_url)

# Convert 'Exchange rate' column to numeric
Dataset['Exchange rate'] = pd.to_numeric(Dataset['Exchange rate'], errors='coerce')

# Remove Rows with empty exchange rate
Dataset = Dataset.dropna(subset=['Exchange rate'])

print(Dataset.info())

# Save preprocessed data
directory = "datasets"

if not os.path.exists(directory) :
    os.makedirs(directory)

filepath = os.path.join(directory, 'preprocessed.csv')

Dataset.to_csv(filepath, index=False)