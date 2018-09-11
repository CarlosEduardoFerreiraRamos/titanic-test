import numpy as np;

import pandas as pd;

import statsmodels.formula.api as sm;

from sklearn.preprocessing import StandardScaler;

from sklearn.linear_model import LogisticRegression;
print("readFiles")
""" READ DATA SET FILES """
df_train = pd.read_csv('data_sets/house_prices_train.csv');
df_test = pd.read_csv('data_sets/house_prices_test.csv');

categotical_columns = ['MSSubClass','MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'RoofStyle', 'RoofMatl', 'Exterior1st',
 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType',
 'SaleCondition'];

for column in  categotical_columns:
	dummies = pd.get_dummies(df_train[column], prefix=column);
	df_train = pd.concat([df_train, dummies], axis = 1);

# print(df_train.columns.values)


