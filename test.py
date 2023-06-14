import pandas as pd

pd.set_option("display.max_columns", None)
df = pd.read_csv("diabetes_prediction_dataset.csv")
print(df.head())
df_slice = df.values[:10000, :]

