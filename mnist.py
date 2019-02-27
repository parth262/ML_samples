import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

train_data = pd.read_csv("resources/mnist.csv")
train_data.head()

X = train_data.drop("label", axis=1)
y = train_data["label"]

pca = PCA(n_components=200)
X_pca = pca.fit_transform(X)

train_X, test_X, train_y, test_y = train_test_split(X_pca, y, test_size=0.3)

clf = RandomForestClassifier()
clf.fit(train_X, train_y)
predictions = clf.predict(test_X)
score = accuracy_score(test_y, predictions)
print(score)
