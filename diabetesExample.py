import numpy as np
import pandas as pd
from sklearn import preprocessing, svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("resources/diabetes.csv", sep=",")
X = np.array(df.drop(['class', 'preg_freq'], 1))
y = np.array(df['class'])

X = preprocessing.scale(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

# KNeighboursClassifier and svm.SVC works great
clf = svm.SVC()
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
print(accuracy_score(y_test, predictions))


