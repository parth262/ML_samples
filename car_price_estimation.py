import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

df = pd.read_csv('resources/car_data/car_data.csv')

df.drop(['losses', 'fuel-system'], 1, inplace=True)
# df['losses'].replace("?", df["losses"].replace("?", 0).astype(float).mean(), inplace=True)

# Replacing unknown values in doors column with the max number of values related to respective company
new_df = df['doors'].groupby(df['company']).agg('max')

unknown_doors_index = df.loc[df['doors'] == '?'].index

for i in unknown_doors_index:
    df.loc[i, 'doors'] = new_df[new_df.index == df['company'].loc[i]][0]

del unknown_doors_index
del new_df

df = df[df['stroke'] != '?']
df = df[df['horsepower'] != '?']
df = df[df['price'] != '?']

company_dict = {"chevrolet": 1, "dodge": 2, "plymouth": 3, "honda": 4, "subaru": 5, "isuzu": 6, "mitsubishi": 7,
                "toyota": 8, "mazda": 9, "volkswagen": 10, "nissan": 11, "saab": 12, "peugot": 13, "alfa-romero": 14,
                "mercury": 15, "audi": 16, "volvo": 17, "bmw": 18, "porsche": 19, "mercedes-benz": 20, "jaguar": 21}

df['company'] = df['company'].map(company_dict)
# df['losses'] = df['losses'].astype('float')
df['bore'] = df['bore'].astype('float')
df['stroke'] = df['stroke'].astype('float')
df['horsepower'] = df['horsepower'].astype('float')
df['peak-rpm'] = df['peak-rpm'].astype('float')
df['price'] = df['price'].astype('float')

df = pd.get_dummies(df, columns=['fuel', 'aspiration', 'doors', 'engine-location'])

# def get_weightage(column):
#     newdf = df['price'].groupby(df[column]).agg('sum')
#     counts = df[column].value_counts()
#     for i in newdf.index:
#         newdf[i] = newdf[i]/counts[i]
#     return newdf.sort_values()


body_mapping = {"hatchback": 1, "wagon": 2, "sedan": 3, "hardtop": 4, "convertible": 5}
dw_mapping = {"fwd": 1, "4wd": 2, "rwd": 3}
et_mapping = {"ohc": 1, "ohcf": 2, "l": 3, "dohc": 4, "ohcv": 5}
cylinders_mapping = {"three": 1, "four": 2, "five": 3, "six": 4, "eight": 5, "twelve": 6}

df['body'] = df['body'].map(body_mapping)
df['drive-wheels'] = df['drive-wheels'].map(dw_mapping)
df['engine-type'] = df['engine-type'].map(et_mapping)
df['cylinders'] = df['cylinders'].map(cylinders_mapping)

df['car-size'] = df['length'] * df['width'] * df['height']
df['bore-stroke-ratio'] = df['bore'] / df['stroke']
df.drop(['length', 'width', 'height', 'bore', 'stroke'], 1, inplace=True)

# corr = df.corr()
# drop_columns = [c for c in corr.columns if corr[c]['price'] < 0]
#
# df.drop(drop_columns, 1, inplace=True)

X = df.drop(['price'], 1)
scalar = StandardScaler()
scalar.fit(X)
X = scalar.transform(X)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y)

clf = xgb.XGBRegressor()

clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
# y_pred = clf.predict(X_test)

print(score)
