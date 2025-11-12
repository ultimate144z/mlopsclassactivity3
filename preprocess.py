import os
import pandas as pd
from sklearn.datasets import load_iris

# Create 'data' directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Load the iris dataset
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target

# Save the preprocessed dataset
data.to_csv("data/preprocessed.csv", index=False)
print("✅ Data preprocessed and saved to data/preprocessed.csv")
