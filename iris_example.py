import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

df = pd.read_csv('resources/iris.txt', sep=",")

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# ax1.scatter(X_train[:, 0], y_train, c='r', marker='*', label="sepal_length", alpha=.5, s=150, linewidths=5)
# ax1.scatter(X_train[:, 1], y_train, c='y', marker='x', label="sepal_width", alpha=.5, s=150, linewidths=5)
# ax1.scatter(X_train[:, 2], y_train, c='b', marker='o', label="petal_length", alpha=.5, s=150, linewidths=5)
# ax1.scatter(X_train[:, 3], y_train, c='k', marker='8', label="petal_length", alpha=.5, s=150, linewidths=5)

labels = {'Iris-setosa': 'r', 'Iris-versicolor': 'b', 'Iris-virginica': 'k'}

[ax1.scatter(X_train[i, 0], X_train[i, 1], X_train[i, 2], c=labels[y_train[i]]) for i in range(len(X_train[:, 0]))]

clf = svm.SVC()
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
print(accuracy_score(y_test, predictions))
plt.legend()
plt.show()
