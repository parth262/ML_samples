from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np

X, y = make_classification(200)

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.1)

clfs = [KNeighborsClassifier(), SVC()]

for clf in clfs:
    clf.fit(train_X, train_y)
    predictions = clf.predict(test_X)
    score = accuracy_score(test_y, predictions)
    cv_score = cross_val_score(clf, X, y)
    print(score)
    print(np.mean(cv_score))
