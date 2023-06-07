import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import knn

# load data
df = pd.read_csv("CS170_small_Data__32.txt", delimiter='  ', header=None, engine='python')
X = df.values[:, 1:]
y = df.values[:, 0]
trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.3)
# scale data
sc = StandardScaler()
scaler = sc.fit(trainX)
trainX_scaled = scaler.transform(trainX)
testX_scaled = scaler.transform(testX)
# perform predictions
k = knn.KNN(k=3)
k.fit(trainX, trainY)
predict = k.predict(testX)
print(accuracy_score(testY, predict))

