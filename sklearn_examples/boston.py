from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

(X, y) = load_boston(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y)

clf = LinearRegression()
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
print(mean_absolute_error(y_test, predictions))

plt.plot(predictions)
plt.plot(y_test)
plt.legend(['predictions', 'actual'])
plt.show()
