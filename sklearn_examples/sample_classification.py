import pandas as pd
from sklearn.tree import DecisionTreeClassifier

X = [{"color": "red", "taste": "sweet", "shape": "almost_round", "label": "apple"},
     {"color": "orange", "taste": "extra_sweet", "shape": "half_moon", "label": "banana"},
     {"color": "orange", "taste": "sour", "shape": "round", "label": "orange"}]

test_X = [{"color": "orange", "taste": "extra_sweet", "shape": "round"}]

color_map = {"red": 0, "yellow": 1, "orange": 2}
taste_map = {"sweet": 0, "sour": 1, "extra_sweet": 2}
shape_map = {"almost_round": 0, "half_moon": 1, "round": 2}

X = pd.DataFrame(X)
test_X = pd.DataFrame(test_X)

X["color"] = X["color"].map(color_map)
X["taste"] = X["taste"].map(taste_map)
X["shape"] = X["shape"].map(shape_map)
test_X["color"] = test_X["color"].map(color_map)
test_X["taste"] = test_X["taste"].map(taste_map)
test_X["shape"] = test_X["shape"].map(shape_map)

clf = DecisionTreeClassifier()
clf.fit(X.drop("label", axis=1), X["label"])
prediction = clf.predict(test_X)
print(prediction)
