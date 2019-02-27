import pandas as pd

train_data = pd.read_csv("resources/predict_future_sales/sales_train_v2.csv")
test_data = pd.read_csv("resources/predict_future_sales/test.csv")
item_data = pd.read_csv("resources/predict_future_sales/items.csv")

date = pd.to_datetime(train_data["date"], format="%d.%m.%Y")
test_data.drop("ID", inplace=True, axis=1)
date.head()

train_data["month"] = date.dt.month
train_data.sample(10)

new_train_data = train_data[train_data["month"] > 8][train_data["month"] <= 11]
new_train_data.head()

# # joining category in train data and test data
id_category_map = {}
for _, row in item_data.iterrows():
    id_category_map.update({row["item_id"]: row["item_category_id"]})
train_data["item_category_id"] = train_data["item_id"].map(id_category_map)
test_data["item_category_id"] = test_data["item_id"].map(id_category_map)
train_data.sample(5)

from sklearn.model_selection import train_test_split

X = train_data[["shop_id", "item_id", "item_category_id"]]
y = train_data["item_cnt_day"]

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3)

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

# clfs = [
#     DecisionTreeRegressor(),
#     RandomForestRegressor(),
#     XGBRegressor()
# ]
# for clf in clfs:
#     clf.fit(train_X, train_y)
#     preds = clf.predict(test_X)
#     score = r2_score(test_y, preds)
#     print(score)

clf = XGBRegressor(n_estimators=500, learning_rate=0.02, n_jobs=2)
clf.fit(train_X, train_y, early_stopping_rounds=5, eval_set=[(test_X, test_y)], verbose=False)
preds = clf.predict(test_X)
score = r2_score(test_y, preds)
print(score)
