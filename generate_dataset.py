# File: generate_dataset.py

import pandas as pd
import numpy as np

# Create a simple dataset
data = {
    'Feature1': np.random.rand(100),
    'Feature2': np.random.rand(100),
    'Feature3': np.random.rand(100),
    'Label': np.random.randint(0, 2, size=100)  # Assuming binary classification
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('./datasets/merged_data.csv', index=False)
