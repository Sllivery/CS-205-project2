import itertools

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from knn import KNN

class ForwardFeatureSelection:
    def __init__(self, k=3):
        self.best_accuracy = None
        self.best_features = None
        self.k = k

    def fit(self, X, y):
        self.best_accuracy = 0
        self.best_features = []
        num_features = X.shape[1]
        combination_indexes = []
        for i in range(num_features):
            combination_indexes.append(i)
        for i in range(num_features):
            for i in itertools.combinations(combination_indexes, i+1):
                temp_features = list(i)
                X_temp = X[:, list(i)]
                accuracy = self.evaluate(X_temp, y)
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    self.best_feature = temp_features
                print("Using feature: " + str(temp_features) + ", accuracy is " + str(accuracy))

    def evaluate(self, X, y):
        trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=42)
        # scale data
        # sc = StandardScaler()
        # scaler = sc.fit(trainX)
        # trainX_scaled = scaler.transform(trainX)
        # testX_scaled = scaler.transform(testX)
        # wrapp the knn
        classifier = KNN(self.k)
        classifier.fit(trainX, trainY)
        y_pred = classifier.predict(testX)
        accuracy = accuracy_score(testY, y_pred, normalize=False)
        return accuracy
