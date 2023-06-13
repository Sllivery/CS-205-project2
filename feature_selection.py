import itertools

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from knn import KNN

class FeatureSelection:
    def __init__(self, k=3, direction="forward", threshold=100):
        self.direction = direction
        self.best_accuracy = None
        self.best_features = None
        self.k = k
        self.last_best_features = []
        self.threshold = threshold
        self.termination_flag = False;

    def fit(self, X, y):
        self.best_accuracy = 0
        self.best_features = []
        num_features = X.shape[1]
        combination_indexes = []
        for i in range(num_features):
            combination_indexes.append(i)
        if self.direction == "forward":
            for i in range(num_features):
                self.selection(combination_indexes, i, X, y)
                if self.termination_flag:
                    return
        elif self.direction == "backward":
            for i in range(num_features, 0, -1):
                self.selection(combination_indexes, i, X, y)
                if self.termination_flag:
                    return

    def selection(self, combination_indexes, i, X, y):
        for i in itertools.combinations(combination_indexes, i + 1):
            temp_features = list(i)
            if self.direction == "forward" and set(temp_features) > set(self.last_best_features): # 如果当前集合是上一次最佳组合的超集，才使用当前的集合继续搜索
                X_temp = X[:, list(i)]
                accuracy = self.evaluate(X_temp, y)
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    self.best_features = temp_features
                print("Using feature: " + str(temp_features) + ", accuracy is " + str(accuracy))
            if self.direction == "backward" and set(self.last_best_features) > set(temp_features):
                X_temp = X[:, list(i)]
                accuracy = self.evaluate(X_temp, y)
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    self.best_features = temp_features
                print("Using feature: " + str(temp_features) + ", accuracy is " + str(accuracy))
            if self.best_accuracy >= self.threshold:
                self.termination_flag = True
                return
        self.last_best_features = self.best_features


    def evaluate(self, X, y):
        trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=42)
        # wrapp the knn
        classifier = KNN(self.k)
        classifier.fit(trainX, trainY)
        y_pred = classifier.predict(testX)
        accuracy = accuracy_score(testY, y_pred)
        return accuracy
