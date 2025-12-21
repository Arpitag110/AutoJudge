import pandas as pd

# Load dataset
df = pd.read_csv("../data/problems.csv")

# Show dataset
print(df)

print("\nDataset info:")
print(df.info())