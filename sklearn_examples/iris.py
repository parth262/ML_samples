from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

iris_data = load_iris()
columns = iris_data.feature_names
X = iris_data.data
y = iris_data.target
labels = iris_data.target_names
X = pd.DataFrame(X, columns=columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

clf = KNeighborsClassifier()
clf.fit(X_train, y_train)

preds = cross_val_predict(clf, X, y, cv=10)

predictions = clf.predict(X_test)

print(accuracy_score(y, preds))
print(accuracy_score(y_test, predictions))
