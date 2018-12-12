from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

(X, y) = load_boston(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

clf = LinearRegression()
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
print(mean_absolute_error(y_test, predictions))
print(r2_score(y_test, predictions))

# plt.plot(predictions)
# plt.plot(y_test)
# plt.legend(['predictions', 'actual'])
# plt.show()

import numpy as np

x_mean = np.mean(X, axis=1)
coeff_mean = np.mean(clf.coef_)

f = coeff_mean * x_mean + clf.intercept_

plt.plot(x_mean, f)
plt.show()
