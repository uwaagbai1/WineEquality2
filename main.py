import numpy as np
import pandas as pd

wine = pd.read_csv('csv/winequality.csv', delimiter=';')
x = wine.iloc[:, 0:-1].values
y = wine.iloc[:, -1].values

#Using KMeans Cluster
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

from sklearn.cluster import KMeans
model = KMeans()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)

table = pd.DataFrame({'Actual': y_test, 'Predicted': y_predict})
print(table.head())
from sklearn.metrics import accuracy_score
acc_KM = accuracy_score(y_test, y_predict) * 100
print(
  'The accuracy gotten from applying the KMeans Clustering Algorithm is',
  round(acc_KM, 2)
, end='% ')