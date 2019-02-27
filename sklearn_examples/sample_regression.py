import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

X = np.array([[1],
              [2],
              [3],
              [4],
              [5],
              [6],
              [7],
              [8],
              [9],
              [10],
              [11],
              [12],
              [13]])
y = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 19, 18])

plt.scatter(X, y)
plt.show()

test_X = np.array([[11],
                   [12],
                   [13],
                   [14],
                   [15],
                   [16],
                   [17]])
test_y = np.array([15, 16, 17, 18, 19, 20, 21])

clf = LinearRegression()
clf.fit(X, y)
preds = clf.predict(test_X)

all_data = np.concatenate((X, test_X))

f = clf.coef_ * all_data + clf.intercept_

plt.scatter(X, y)
plt.plot(all_data, f)
plt.show()
