from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

centers = [(-5, -5), (0, 0), (5, 5)]
X, y = make_blobs(500, random_state=42, shuffle=False, centers=centers)
plt.scatter(X[:, 0], X[:, 1])

clf = KMeans(n_clusters=6)
clf.fit(X)
cc = clf.cluster_centers_
plt.scatter(cc[:, 0], cc[:, 1], s=200)
plt.show()
