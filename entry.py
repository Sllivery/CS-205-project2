import pandas as pd
from feature_selection import FeatureSelection

file_name = ""

print("Welcome to Siwei&Chuanye Fearture Selection Algorithm")
while 1:
    file_setting = input("Select the file to test: 1)samll 2)large 3)xlarge")
    if file_setting == "1":
        file_name = "CS170_small_Data__25.txt"
        break
    elif file_setting == "2":
        file_name = "CS170_large_Data__16.txt"
        break
    elif file_setting == "3":
        file_name = "CS170_XXXlarge_Data__12.txt"
        break
    else:
        print("Input error, please use 1 or 2 or 3!")

# load data
df = pd.read_csv(file_name, delimiter='  ', header=None, engine='python')
X = df.values[:, 1:]
y = df.values[:, 0]

# perform the algorithm
while 1:
    algorithm_setting = input("Type the algorithm number you want to run: 1)ForwardSelection 2)Backward Elimination")
    if algorithm_setting == "1": # 1 and default are forward
        # perform predictions
        k = FeatureSelection(k=3)
        k.fit(X, y)
        print(k.best_accuracy)
        print(k.best_features)
        break
    elif algorithm_setting == "2": # 2 is backward
        k = FeatureSelection(k=3, direction="backward")
        k.fit(X, y)
        print(k.best_accuracy)
        print(k.best_features)
        break
    else:
        print("Input error, please use 1 or 2!")


