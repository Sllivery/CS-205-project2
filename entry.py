import datetime
import pandas as pd
from sklearn.preprocessing import StandardScaler

from feature_selection import FeatureSelection

file_name = ""

def load_test_data():
    # load data
    df = pd.read_csv(file_name, delimiter='  ', header=None, engine='python')
    X = df.values[:, 1:]
    y = df.values[:, 0]
    return X, y

def load_real_world_data():
    df = pd.read_csv(file_name, nrows=1000)
    df.drop('HbA1c_level', axis=1, inplace=True)
    df.drop(df[df['smoking_history'] == 'No Info'].index, inplace=True)
    df['gender'].replace({'Female': 0, 'Male': 1}, inplace=True)
    df['smoking_history'].replace({'never': 0, 'current': 1, 'former': 2, 'not current': 3, 'ever': 4}, inplace=True)
    X = df.values[:, :-1]
    y = df.values[:, -1]
    sc = StandardScaler()
    sc.fit(X)
    X_scaled = sc.transform(X)
    return X_scaled, y

print("Welcome to Siwei&Chuanye Fearture Selection Algorithm")
while 1:
    file_setting = input("Select the file to test: 1)samll 2)large 3)xlarge 4)real_world:diabetes_prediction")
    if file_setting == "1":
        file_name = "CS170_small_Data__25.txt"
        X,y = load_test_data()
        break
    elif file_setting == "2":
        file_name = "CS170_large_Data__16.txt"
        X,y = load_test_data()
        break
    elif file_setting == "3":
        file_name = "CS170_XXXlarge_Data__12.txt"
        X,y = load_test_data()
        break
    elif file_setting == "4":
        file_name = "diabetes_prediction_dataset.csv"
        X,y = load_real_world_data()
        break
    else:
        print("Input error, please use 1 or 2 or 3 or 4 !")


# perform the algorithm
while 1:
    algorithm_setting = input("Type the algorithm number you want to run: 1)ForwardSelection 2)Backward Elimination")
    threshold = input("Type the threshold that you want to end the program early (from 0 to 1")
    if algorithm_setting == "1": # 1 and default are forward
        # perform predictions
        start = datetime.datetime.now()
        k = FeatureSelection(k=3, threshold=float(threshold))
        k.fit(X, y)
        print(k.best_accuracy)
        print(k.best_features)
        end = datetime.datetime.now()
        print((end - start).seconds)
        break
    elif algorithm_setting == "2": # 2 is backward
        start = datetime.datetime.now()
        k = FeatureSelection(k=3, direction="backward", threshold=float(threshold))
        k.fit(X, y)
        print(k.best_accuracy)
        print(k.best_features)
        end = datetime.datetime.now()
        print((end - start).seconds)
        break
    else:
        print("Input error, please use 1 or 2!")


