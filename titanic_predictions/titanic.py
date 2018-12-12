import re

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier

train_data = pd.read_csv("resources/titanic/train.csv")
test_data = pd.read_csv("resources/titanic/test.csv")
full_data = [train_data, test_data]

test_passenger_ids = test_data["PassengerId"]


def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""


for dataset in full_data:
    dataset['FamilySize'] = dataset["SibSp"] + dataset["Parch"] + 1
    dataset['IsAlone'] = dataset["FamilySize"].apply(lambda x: 1 if x == 1 else 0)
    dataset["Embarked"].fillna("S", inplace=True)
    dataset["Fare"].fillna(train_data["Fare"].median(), inplace=True)
    age_avg = dataset["Age"].mean()
    age_std = dataset["Age"].std()
    age_null_count = dataset["Age"].isnull().sum()
    random_age_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset["Age"][np.isnan(dataset["Age"])] = random_age_list
    dataset["Age"] = dataset["Age"].astype(int)
    dataset["Title"] = dataset["Name"].apply(get_title)
    dataset['Title'] = dataset['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
train_data["CategoricalFare"] = pd.qcut(train_data["Fare"], 4)
train_data["CategoricalAge"] = pd.cut(train_data["Age"], 5)


def map_age(age):
    if age <= 16:
        return 0
    elif 16 < age <= 32:
        return 1
    elif 32 < age <= 48:
        return 2
    elif 48 < age <= 64:
        return 3
    else:
        return 4


def map_fare(fare):
    if fare <= 7.91:
        return 0
    elif 7.91 < fare <= 14.454:
        return 1
    elif 14.454 < fare <= 31:
        return 2
    else:
        return 3


for dataset in full_data:
    # Mapping Sex
    dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)

    # Mapping titles
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

    # Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    # Mapping Fare
    dataset["Fare"] = dataset["Fare"].apply(map_fare)
    dataset['Fare'] = dataset['Fare'].astype(int)

    # Mapping Age
    dataset["Age"] = dataset["Age"].apply(map_age)

drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'FamilySize']
train_data = train_data.drop(drop_elements, axis=1)
train_data = train_data.drop(['CategoricalAge', 'CategoricalFare'], axis=1)

test_data = test_data.drop(drop_elements, axis=1)

X = train_data.drop("Survived", axis=1)
y = train_data["Survived"]

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.1)

clfs = [SVC(), RandomForestClassifier(), ExtraTreesClassifier(), AdaBoostClassifier(), GradientBoostingClassifier()]

for clf in clfs:
    clf.fit(train_X, train_y)
    predictions = clf.predict(test_X)
    score = accuracy_score(test_y, predictions)
    print(score)

final_clf = SVC()
final_clf.fit(X, y)
final_predictions = final_clf.predict(test_data)

output = pd.DataFrame({"PassengerId": test_passenger_ids, "Survived": final_predictions})
output.to_csv("resources/titanic/submission.csv", index=False)
