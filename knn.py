import numpy as np

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        y_pred = []
        for x in X:
            distances = []
            for i in range(len(self.X_train)):
                distance = self.euclidean_distance(x, self.X_train[i])
                distances.append((distance, self.y_train[i]))
            distances.sort(key=lambda x: x[0])
            k_nearest_neighbors = distances[:self.k]
            labels = [neighbor[1] for neighbor in k_nearest_neighbors]
            y_pred.append(max(set(labels), key=labels.count))
        return y_pred