import numpy as np;
import pandas as pd;
import statsmodels.formula.api as sm;
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler;
from sklearn.linear_model import LogisticRegression;
from sklearn.preprocessing import StandardScaler;
from sklearn.preprocessing import Imputer;
from sklearn.linear_model import LinearRegression;

""" READ DATA SET FILES """
df_train = pd.read_csv('data_sets/house_prices_train.csv');
# df_test = pd.read_csv('data_sets/house_prices_test.csv');

# hold test id for later
# test_ids = df_test['Id']; 
train_set_number = len(df_train)
# del df_train['Id'];
# del df_test['Id'];

""" CREATE DUMMY VARIABLES """
categotical_columns = ['MSSubClass','MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'RoofStyle', 'RoofMatl', 'Exterior1st',
 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType',
 'SaleCondition'];

# concat train and test sets to creat the same dummy variables
concateneted_dateset = df_train;

for column in  categotical_columns:
	# replace categotical missing data
	concateneted_dateset[column] = concateneted_dateset[column].apply(lambda x: 'MISSING' if str(x) == 'nan'  else x);
	# create dummy variables
	dummies = pd.get_dummies(concateneted_dateset[column], prefix=column);
	concateneted_dateset = pd.concat([concateneted_dateset, dummies], axis = 1);

# split the dataset back to train and test sets
trainIndexs = np.random.rand(len(concateneted_dateset)) < 0.8
df_train = concateneted_dateset[trainIndexs];
df_test = concateneted_dateset[~trainIndexs];
train_ids = df_train["Id"]
test_ids = df_test["Id"] 

"""REMOVE ORIGINAL CATEGORICAL COLUMNS"""
df_train = df_train.drop(columns = categotical_columns);
df_test = df_test.drop(columns = categotical_columns);

"""REPLACE NUMERICAL MISSING VALUES"""
columns_train = df_train.columns[df_train.isnull().any()].tolist();
imputer = Imputer(strategy = 'mean', missing_values = 'NaN', axis = 0);
imputer.fit(df_train[columns_train]);
df_train[columns_train] = imputer.transform(df_train[columns_train])

columns_test = df_test.columns[df_test.isnull().any()].tolist();
imputer = Imputer(strategy = 'mean', missing_values = 'NaN', axis = 0);
imputer.fit(df_test[columns_test]);
df_test[columns_test] = imputer.transform(df_test[columns_test])

test_columns = list(df_test.columns.values)
""" FUTURE SCALING THE DATA"""
fitted = StandardScaler().fit(df_train.loc[:, df_train.columns != "SalePrice"]);
df_train.loc[:, df_train.columns != "SalePrice"] = fitted.transform(df_train.loc[:, df_train.columns != "SalePrice"]);
df_test.loc[:, df_train.columns != "SalePrice"] = fitted.transform(df_test.loc[:, df_test.columns != "SalePrice"]);

"""ADDING b0"""
data_length = len(df_train);
df_train['b0'] = [1]*data_length;

"""BACKWARD ELIMINATION"""
max_p_value = 1;
non_significant_column = None;
eliminator = None;
num = 0
sm_result = None
while max_p_value > 0.05:
	if not non_significant_column == None:
		del df_train[non_significant_column];
		del df_test[non_significant_column];
	sm_result = sm.OLS(endog = df_train["SalePrice"], exog = df_train.loc[:, df_train.columns != "SalePrice"]).fit();
	p_values = sm_result.pvalues;
	max_p_value = np.amax(p_values)
	i = np.where(p_values == max_p_value);
	non_significant_column = list(p_values.index[i])[0];

# remove b0
del df_train['b0'];
""" LOGISTIC REGRESSION """
regressor = LogisticRegression(random_state=0);
regressor.fit(df_train.loc[:, df_train.columns != "SalePrice"], df_train["SalePrice"]);
prediction = regressor.predict(df_test.loc[:, df_test.columns != "SalePrice"]);

# #################
# SUBMIT ANSWER
# #################

plt.scatter(test_ids, df_test["SalePrice"], color = 'red')
plt.plot(test_ids, prediction, color = 'blue')
plt.
plt.title('Linear ')
plt.xlabel('Id')
plt.ylabel('Sale Price')
plt.show()



# plt.hist(list(df_test["SalePrice"]))
# plt.title('Histogram of ads selections')
# plt.xlabel('Ads')
# plt.ylabel('Number of times each ad was selected')
# plt.show()


sub_df = {
	"Id":test_ids,
	"salesPrice": df_test["SalePrice"],
	"Predict":  prediction	
};

ds = pd.DataFrame(sub_df);
ds.to_csv("house_prices_submission_test.csv", index=False);



 


