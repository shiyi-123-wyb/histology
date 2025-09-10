import pandas as pd

df = pd.read_csv('/media/public506/sdb/SY/10k/splits_0.csv')
print("Total rows:", len(df))
print("Train samples:", df['train'].notna().sum())
print("Val samples:", df['val'].notna().sum())
print("Test samples:", df['test'].notna().sum())
print("\nSample of the data:")
print(df.head())