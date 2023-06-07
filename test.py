import pandas as pd

# Create a DataFrame from a dictionary
data = {'Name': ['John', 'Emma', 'Michael'],
        'Age': [25, 28, 32],
        'City': ['New York', 'London', 'Paris']}
df = pd.DataFrame(data)
print(df.values[:, [0,2]])
