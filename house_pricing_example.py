# Loading libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.stats import skew
import xgboost as xgb
from sklearn.linear_model import Lasso

train = pd.read_csv('resources/house_pricing/train.csv')
test = pd.read_csv('resources/house_pricing/test.csv')

train.drop(train[train['GrLivArea'] > 4000].index, inplace=True)

all_data = train.append(test, sort=False)

all_data.drop(['Id'], inplace=True, axis=1)

# Data Preprocessing
le = LabelEncoder()


def encode_labels(data, column, replacement=None):
    if replacement is not None:
        data[column].fillna(replacement, inplace=True)
    le.fit(data[column])
    data[column] = le.transform(data[column])
    return data


lot_frontage_by_neighborhood = train['LotFrontage'].groupby(train['Neighborhood'])

for key, group in lot_frontage_by_neighborhood:
    idx = (all_data['Neighborhood'] == key) & (all_data['LotFrontage'].isnull())
    all_data.loc[idx, 'LotFrontage'] = group.median()

all_data["MasVnrArea"].fillna(0, inplace=True)
all_data["BsmtFinSF1"].fillna(0, inplace=True)
all_data["BsmtFinSF2"].fillna(0, inplace=True)
all_data["BsmtUnfSF"].fillna(0, inplace=True)
all_data["TotalBsmtSF"].fillna(0, inplace=True)
all_data["GarageArea"].fillna(0, inplace=True)
all_data["BsmtFullBath"].fillna(0, inplace=True)
all_data["BsmtHalfBath"].fillna(0, inplace=True)
all_data["GarageCars"].fillna(0, inplace=True)
all_data["GarageYrBlt"].fillna(0.0, inplace=True)
all_data["PoolArea"].fillna(0, inplace=True)

qual_dict = {np.nan: 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
names = np.array(['ExterQual', 'PoolQC', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu',
                  'GarageQual', 'GarageCond'])

for name in names:
    all_data[name] = all_data[name].map(qual_dict).astype(int)

all_data["BsmtExposure"] = all_data["BsmtExposure"].map({np.nan: 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}).astype(int)

bsmt_fin_dict = {np.nan: 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}
all_data["BsmtFinType1"] = all_data["BsmtFinType1"].map(bsmt_fin_dict).astype(int)
all_data["BsmtFinType2"] = all_data["BsmtFinType2"].map(bsmt_fin_dict).astype(int)

all_data["Functional"] = all_data["Functional"].map(
    {np.nan: 0, "Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8}).astype(int)
all_data["GarageFinish"] = all_data["GarageFinish"].map({np.nan: 0, "Unf": 1, "RFn": 2, "Fin": 3}).astype(int)
all_data["Fence"] = all_data["Fence"].map({np.nan: 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4}).astype(int)

all_data["CentralAir"] = (all_data["CentralAir"] == "Y") * 1.0
varst = np.array(
    ['MSSubClass', 'LotConfig', 'Neighborhood', 'Condition1', 'BldgType', 'HouseStyle', 'RoofStyle', 'Foundation',
     'SaleCondition'])

for x in varst:
    encode_labels(all_data, x)

all_data = encode_labels(all_data, "MSZoning", "RL")
all_data = encode_labels(all_data, "Exterior1st", "Other")
all_data = encode_labels(all_data, "Exterior2nd", "Other")
all_data = encode_labels(all_data, "MasVnrType", "None")
all_data = encode_labels(all_data, "SaleType", "Oth")

# Feature Engineering

all_data["IsRegularLotShape"] = (all_data["LotShape"] == "Reg") * 1
all_data["IsLandLevel"] = (all_data["LandContour"] == "Lvl") * 1
all_data["IsLandSlopeGentle"] = (all_data["LandSlope"] == "Gtl") * 1
all_data["IsElectricalSBrkr"] = (all_data["Electrical"] == "SBrkr") * 1
all_data["IsGarageDetached"] = (all_data["GarageType"] == "Detchd") * 1
all_data["IsPavedDrive"] = (all_data["PavedDrive"] == "Y") * 1
all_data["HasShed"] = (all_data["MiscFeature"] == "Shed") * 1
all_data["Remodeled"] = (all_data["YearRemodAdd"] != all_data["YearBuilt"]) * 1
all_data["RecentRemodel"] = (all_data["YearRemodAdd"] == all_data["YrSold"]) * 1
all_data["VeryNewHouse"] = (all_data["YearBuilt"] == all_data["YrSold"]) * 1
all_data["Has2ndFloor"] = (all_data["2ndFlrSF"] == 0) * 1
all_data["HasMasVnr"] = (all_data["MasVnrArea"] == 0) * 1
all_data["HasWoodDeck"] = (all_data["WoodDeckSF"] == 0) * 1
all_data["HasOpenPorch"] = (all_data["OpenPorchSF"] == 0) * 1
all_data["HasEnclosedPorch"] = (all_data["EnclosedPorch"] == 0) * 1
all_data["Has3SsnPorch"] = (all_data["3SsnPorch"] == 0) * 1
all_data["HasScreenPorch"] = (all_data["ScreenPorch"] == 0) * 1

all_data["HighSeason"] = all_data["MoSold"].replace(
    {1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0})
all_data["NewerDwelling"] = all_data["MSSubClass"].replace(
    {20: 1, 30: 0, 40: 0, 45: 0, 50: 0, 60: 1, 70: 0, 75: 0, 80: 0, 85: 0, 90: 0, 120: 1, 150: 0, 160: 0, 180: 0,
     190: 0})

all_data2 = train.append(test, sort=False)

all_data["SaleCondition_PriceDown"] = all_data2.SaleCondition.replace(
    {'Abnorml': 1, 'Alloca': 1, 'AdjLand': 1, 'Family': 1, 'Normal': 0, 'Partial': 0})
all_data["BoughtOffPlan"] = all_data2.SaleCondition.replace(
    {"Abnorml": 0, "Alloca": 0, "AdjLand": 0, "Family": 0, "Normal": 0, "Partial": 1})
all_data["BadHeating"] = all_data2.HeatingQC.replace({'Ex': 0, 'Gd': 0, 'TA': 0, 'Fa': 1, 'Po': 1})

area_cols = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
             '2ndFlrSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
             'ScreenPorch', 'LowQualFinSF', 'PoolArea']

all_data["TotalArea"] = all_data[area_cols].sum(axis=1)
all_data["TotalArea1st2nd"] = all_data["1stFlrSF"] + all_data["2ndFlrSF"]
all_data["Age"] = 2010 - all_data["YearBuilt"]
all_data["TimeSinceSold"] = 2010 - all_data["YrSold"]
all_data["SeasonSold"] = all_data["MoSold"].map(
    {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}).astype(int)
all_data["YearsSinceRemodel"] = all_data["YrSold"] - all_data["YearRemodAdd"]

# Simplifications of existing features into bad/average/good based on counts
all_data["SimplOverallQual"] = all_data.OverallQual.replace(
    {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 3})
all_data["SimplOverallCond"] = all_data.OverallCond.replace(
    {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 3})
all_data["SimplPoolQC"] = all_data.PoolQC.replace({1: 1, 2: 1, 3: 2, 4: 2})
all_data["SimplGarageCond"] = all_data.GarageCond.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
all_data["SimplGarageQual"] = all_data.GarageQual.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
all_data["SimplFireplaceQu"] = all_data.FireplaceQu.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
all_data["SimplFireplaceQu"] = all_data.FireplaceQu.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
all_data["SimplFunctional"] = all_data.Functional.replace({1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 3, 8: 4})
all_data["SimplKitchenQual"] = all_data.KitchenQual.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
all_data["SimplHeatingQC"] = all_data.HeatingQC.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
all_data["SimplBsmtFinType1"] = all_data.BsmtFinType1.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2})
all_data["SimplBsmtFinType2"] = all_data.BsmtFinType2.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2})
all_data["SimplBsmtCond"] = all_data.BsmtCond.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
all_data["SimplBsmtQual"] = all_data.BsmtQual.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
all_data["SimplExterCond"] = all_data.ExterCond.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
all_data["SimplExterQual"] = all_data.ExterQual.replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 2})

neighborhood_map = {"MeadowV": 0, "IDOTRR": 1, "BrDale": 1, "OldTown": 1, "Edwards": 1, "BrkSide": 1, "Sawyer": 1,
                    "Blueste": 1, "SWISU": 2, "NAmes": 2, "NPkVill": 2, "Mitchel": 2, "SawyerW": 2, "Gilbert": 2,
                    "NWAmes": 2, "Blmngtn": 2, "CollgCr": 2, "ClearCr": 3, "Crawfor": 3, "Veenker": 3, "Somerst": 3,
                    "Timber": 3, "StoneBr": 4, "NoRidge": 4, "NridgHt": 4}

all_data['NeighborhoodBin'] = all_data2['Neighborhood'].map(neighborhood_map)
all_data.loc[all_data2.Neighborhood == 'NridgHt', "Neighborhood_Good"] = 1
all_data.loc[all_data2.Neighborhood == 'Crawfor', "Neighborhood_Good"] = 1
all_data.loc[all_data2.Neighborhood == 'StoneBr', "Neighborhood_Good"] = 1
all_data.loc[all_data2.Neighborhood == 'Somerst', "Neighborhood_Good"] = 1
all_data.loc[all_data2.Neighborhood == 'NoRidge', "Neighborhood_Good"] = 1
all_data["Neighborhood_Good"].fillna(0, inplace=True)
all_data["SaleCondition_PriceDown"] = all_data2.SaleCondition.replace(
    {'Abnorml': 1, 'Alloca': 1, 'AdjLand': 1, 'Family': 1, 'Normal': 0, 'Partial': 0})

# House completed before sale or not
all_data["BoughtOffPlan"] = all_data2.SaleCondition.replace(
    {"Abnorml": 0, "Alloca": 0, "AdjLand": 0, "Family": 0, "Normal": 0, "Partial": 1})
all_data["BadHeating"] = all_data2.HeatingQC.replace({'Ex': 0, 'Gd': 0, 'TA': 0, 'Fa': 1, 'Po': 1})

train_new = all_data[all_data['SalePrice'].notnull()]
test_new = all_data[all_data['SalePrice'].isnull()]

numeric_features = [f for f in train_new.columns if train_new[f].dtype != object]

skewed = train_new[numeric_features].apply(lambda x: skew(x.dropna().astype(float)))
skewed = skewed[skewed > 0.75]
skewed = skewed.index
train_new[skewed] = np.log1p(train_new[skewed])
test_new[skewed] = np.log1p(test_new[skewed])
del test_new['SalePrice']

scaler = StandardScaler()
scaler.fit(train_new[numeric_features])
scaled = scaler.transform(train_new[numeric_features])

for i, col in enumerate(numeric_features):
    train_new[col] = scaled[:, i]

numeric_features.remove('SalePrice')

scaled = scaler.fit_transform(test_new[numeric_features])

for i, col in enumerate(numeric_features):
    test_new[col] = scaled[:, i]


def onehot(onehot_data, data, column, fill_na):
    onehot_data[column] = data[column]
    if fill_na is not None:
        onehot_data[column].fillna(fill_na, inplace=True)
    dummies = pd.get_dummies(onehot_data[column], prefix="_" + column)
    onehot_data = onehot_data.join(dummies)
    onehot_data.drop([column], inplace=True, axis=1)
    return onehot_data


def munge_onehot(df):
    onehot_df = pd.DataFrame(index=df.index)

    onehot_df = onehot(onehot_df, df, "MSSubClass", None)
    onehot_df = onehot(onehot_df, df, "MSZoning", "RL")
    onehot_df = onehot(onehot_df, df, "LotConfig", None)
    onehot_df = onehot(onehot_df, df, "Neighborhood", None)
    onehot_df = onehot(onehot_df, df, "Condition1", None)
    onehot_df = onehot(onehot_df, df, "BldgType", None)
    onehot_df = onehot(onehot_df, df, "HouseStyle", None)
    onehot_df = onehot(onehot_df, df, "RoofStyle", None)
    onehot_df = onehot(onehot_df, df, "Exterior1st", "VinylSd")
    onehot_df = onehot(onehot_df, df, "Exterior2nd", "VinylSd")
    onehot_df = onehot(onehot_df, df, "Foundation", None)
    onehot_df = onehot(onehot_df, df, "SaleType", "WD")
    onehot_df = onehot(onehot_df, df, "SaleCondition", "Normal")

    # Fill in missing MasVnrType for rows that do have a MasVnrArea.
    temp_df = df[["MasVnrType", "MasVnrArea"]].copy()
    idx = (df["MasVnrArea"] != 0) & ((df["MasVnrType"] == "None") | (df["MasVnrType"].isnull()))
    temp_df.loc[idx, "MasVnrType"] = "BrkFace"
    onehot_df = onehot(onehot_df, temp_df, "MasVnrType", "None")

    onehot_df = onehot(onehot_df, df, "LotShape", None)
    onehot_df = onehot(onehot_df, df, "LandContour", None)
    onehot_df = onehot(onehot_df, df, "LandSlope", None)
    onehot_df = onehot(onehot_df, df, "Electrical", "SBrkr")
    onehot_df = onehot(onehot_df, df, "GarageType", "None")
    onehot_df = onehot(onehot_df, df, "PavedDrive", None)
    onehot_df = onehot(onehot_df, df, "MiscFeature", "None")
    onehot_df = onehot(onehot_df, df, "Street", None)
    onehot_df = onehot(onehot_df, df, "Alley", "None")
    onehot_df = onehot(onehot_df, df, "Condition2", None)
    onehot_df = onehot(onehot_df, df, "RoofMatl", None)
    onehot_df = onehot(onehot_df, df, "Heating", None)

    # we'll have these as numerical variables too
    onehot_df = onehot(onehot_df, df, "ExterQual", "None")
    onehot_df = onehot(onehot_df, df, "ExterCond", "None")
    onehot_df = onehot(onehot_df, df, "BsmtQual", "None")
    onehot_df = onehot(onehot_df, df, "BsmtCond", "None")
    onehot_df = onehot(onehot_df, df, "HeatingQC", "None")
    onehot_df = onehot(onehot_df, df, "KitchenQual", "TA")
    onehot_df = onehot(onehot_df, df, "FireplaceQu", "None")
    onehot_df = onehot(onehot_df, df, "GarageQual", "None")
    onehot_df = onehot(onehot_df, df, "GarageCond", "None")
    onehot_df = onehot(onehot_df, df, "PoolQC", "None")
    onehot_df = onehot(onehot_df, df, "BsmtExposure", "None")
    onehot_df = onehot(onehot_df, df, "BsmtFinType1", "None")
    onehot_df = onehot(onehot_df, df, "BsmtFinType2", "None")
    onehot_df = onehot(onehot_df, df, "Functional", "Typ")
    onehot_df = onehot(onehot_df, df, "GarageFinish", "None")
    onehot_df = onehot(onehot_df, df, "Fence", "None")
    onehot_df = onehot(onehot_df, df, "MoSold", None)

    # Divide  the years between 1871 and 2010 into slices of 20 years
    year_map = pd.concat(
        pd.Series("YearBin" + str(i + 1), index=range(1871 + i * 20, 1891 + i * 20)) for i in range(0, 7))
    yearbin_df = pd.DataFrame(index=df.index)
    yearbin_df["GarageYrBltBin"] = df.GarageYrBlt.map(year_map)
    yearbin_df["GarageYrBltBin"].fillna("NoGarage", inplace=True)
    yearbin_df["YearBuiltBin"] = df.YearBuilt.map(year_map)
    yearbin_df["YearRemodAddBin"] = df.YearRemodAdd.map(year_map)

    onehot_df = onehot(onehot_df, yearbin_df, "GarageYrBltBin", None)
    onehot_df = onehot(onehot_df, yearbin_df, "YearBuiltBin", None)
    onehot_df = onehot(onehot_df, yearbin_df, "YearRemodAddBin", None)
    return onehot_df


onehot_df = munge_onehot(train)

neighborhood_train = pd.DataFrame(index=train_new.shape)
neighborhood_train['NeighborhoodBin'] = train_new['NeighborhoodBin']
neighborhood_test = pd.DataFrame(index=test_new.shape)
neighborhood_test['NeighborhoodBin'] = test_new['NeighborhoodBin']

onehot_df = onehot(onehot_df, neighborhood_train, 'NeighborhoodBin', None)

train_new = train_new.join(onehot_df)

onehot_df_te = munge_onehot(test)
onehot_df_te = onehot(onehot_df_te, neighborhood_test, "NeighborhoodBin", None)
test_new = test_new.join(onehot_df_te)

drop_cols = ["_Exterior1st_ImStucc", "_NeighborhoodBin_-1.1306920120473076",
             "_NeighborhoodBin_-0.04760808471778121", "_Exterior1st_Stone", "_Exterior2nd_Other", "_HouseStyle_2.5Fin",
             "_RoofMatl_Membran", "_RoofMatl_Metal", "_RoofMatl_Roll", "_Condition2_RRAe", "_Condition2_RRAn",
             "_Condition2_RRNn", "_Heating_Floor", "_Heating_OthW", "_Electrical_Mix", "_MiscFeature_TenC",
             "_GarageQual_Ex", "_PoolQC_Fa"]
train_new.drop(drop_cols, axis=1, inplace=True)

test_new.drop(["_MSSubClass_150", "_NeighborhoodBin_-0.026189140043946093"], axis=1, inplace=True)

# Drop these columns
drop_cols = ["_Condition2_PosN",  # only two are not zero
             "_MSZoning_C (all)",
             "_MSSubClass_160"]

train_new.drop(drop_cols, axis=1, inplace=True)
test_new.drop(drop_cols, axis=1, inplace=True)

cat_features = ['Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LandSlope',
                'Condition2', 'RoofMatl', 'Heating', 'Electrical', 'GarageType',
                'PavedDrive', 'MiscFeature']

cat_data = train_new.select_dtypes(exclude=[np.number])

train_new = train_new.select_dtypes(include=[np.number])
test_new = test_new.select_dtypes(include=[np.number])

train_set = train_new.drop(['SalePrice'], axis=1)

label_df = pd.DataFrame(index=train_new.index, columns=['SalePrice'])
label_df['SalePrice'] = np.array(np.log(train['SalePrice']))

# X_train, X_test, y_train, y_test = train_test_split(train_new, label_df, test_size=.2)

XGB = xgb.XGBRegressor(colsample_bytree=0.2,
                       gamma=0.0,
                       learning_rate=0.05,
                       max_depth=6,
                       min_child_weight=1,
                       n_estimators=7200,
                       reg_alpha=0.9,
                       reg_lambda=0.6,
                       subsample=0.2,
                       seed=42,
                       silent=True)

XGB.fit(train_set, label_df)

XGB_pred = XGB.predict(test_new)

las = Lasso()

las.fit(train_set, label_df)

las_pred = las.predict(test_new)

y_pred = (XGB_pred + las_pred) / 2
y_pred = np.exp(y_pred)
pred_df = pd.DataFrame(y_pred, index=test["Id"], columns=["SalePrice"])
# pred_df.to_csv('ensemble1.csv', header=True, index_label='Id')
