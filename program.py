import pandas as pd
import numpy as np

data = pd.read_csv("Dataset1.csv")
# Remove the 'Class' column
# data_cleaned = data.drop(columns=['Class'])

# Display the cleaned data
data.head()
