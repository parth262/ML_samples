from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

(X, y) = load_iris(True)

X_train, X_test, y_train, y_test = train_test_split(X, y)

clf = KNeighborsClassifier()
clf.fit(X_train, y_train)

preds = cross_val_predict(clf, X, y, cv=10)

predictions = clf.predict(X_test)

print(accuracy_score(y, preds))
print(accuracy_score(y_test, predictions))
