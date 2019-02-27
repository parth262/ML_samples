import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

df = pd.read_csv("resources/black_friday.csv")

# df.groupby("Occupation")["Purchase"].sum().plot(kind="bar")
# df.groupby("City_Category")["Purchase"].sum().plot(kind="pie", autopct="%1.1f%%", startangle=45, explode=(0.0, 0.1, 0.0), shadow=True)
# df.groupby("Gender")["Purchase"].sum().plot(kind="bar")
# To combine gender and marital status
df["Combined_G_M"] = df.apply(lambda x: "%s_%s" %(x["Gender"], x["Marital_Status"]), axis=1)
# df.groupby("Combined_G_M")["Purchase"].sum().plot(kind="bar")
# df.groupby("Stay_In_Current_City_Years")["Purchase"].sum().plot(kind="bar")
categories = ["Product_Category_1", "Product_Category_2", "Product_Category_3"]
for category in categories:
    df.groupby(category)["Purchase"].sum().plot(kind="bar")
    plt.show()



def get_encoded_value(data):
    le = LabelEncoder()
    le.fit(data)
    return le.transform(data)


def product_category():
    new_df = df.drop(
        columns=["City_Category", "Product_Category_2", "Product_Category_3", "Stay_In_Current_City_Years", "Purchase",
                 "Product_ID",
                 "User_ID"])
    # city_category_mapping = {"A": 0, "B": 1, "C": 2}
    gender_mapping = {"M": 0, "F": 1}
    age_mapping = {"0-17": 0, "46-50": 1, "26-35": 2, "51-55": 3, "36-45": 4, "18-25": 5, "55+": 6}

    # new_df["City_Category"] = new_df["City_Category"].map(city_category_mapping)
    new_df["Gender"] = new_df["Gender"].map(gender_mapping)
    new_df["Age"] = new_df["Age"].map(age_mapping)

    x = new_df.drop(columns=["Product_Category_1"])
    y = new_df["Product_Category_1"].astype("float")

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.20)

    train_and_test(x_train, x_test, y_train, y_test)


def train_and_test(x_train, x_test, y_train, y_test, depth=100):
    clfs = [
        ("KNN", KNeighborsClassifier(n_neighbors=8)),
        # ("SVC2", SVC(gamma=2)),
        # ("SVC", SVC(kernel="linear", C=0.025)),
        ("MLP", MLPClassifier()),
        ("DecisionTree", DecisionTreeClassifier(max_leaf_nodes=depth)),
        ("GNB", GaussianNB())
    ]
    for name, clf in clfs:
        clf.fit(x_train, y_train)
        y_predict = clf.predict(x_test)
        score = accuracy_score(y_test, y_predict)
        print(name + "=> " + str(score))


def gender(depth=100):
    new_df = df.drop(
        columns=["City_Category", "Combined_G_M", "Stay_In_Current_City_Years", "Purchase", "Product_ID",
                 "User_ID"])
    gender_mapping = {"M": 0, "F": 1}
    age_mapping = {"0-17": 0, "46-50": 1, "26-35": 2, "51-55": 3, "36-45": 4, "18-25": 5, "55+": 6}
    new_df["Age"] = new_df["Age"].map(age_mapping)
    new_df["Gender"] = new_df["Gender"].map(gender_mapping)
    # new_df["Product_ID"] = get_encoded_value(new_df["Product_ID"])
    x = new_df.drop(columns=["Gender"])
    y = new_df["Gender"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    train_and_test(x_train, x_test, y_train, y_test, depth)
