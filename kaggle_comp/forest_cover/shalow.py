import numpy as np
import pandas as pd
import statsmodels.formula.api as sm;
import matplotlib.pyplot as plt

from plot import Plot
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

""" READ DATA SET FILES """
# df_train = pd.read_csv('../input/train.csv');
# df_test = pd.read_csv('../input/test.csv');
df_train = pd.read_csv('./data_sets/forest_cover_types_train.csv');
df_test = pd.read_csv('./data_sets/forest_cover_types_test.csv');
dependent_varible = "Cover_Type"
index_varible = "Id"

""" DATA FROM DATAFRAME TO ARRAY"""
X_train = df_train.loc[:, df_train.columns != dependent_varible]
y_train = df_train[dependent_varible]
X_test = df_test

# split the dataset back to train and test sets
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(df_train.loc[:, df_train.columns != dependent_varible],
# 													df_train[dependent_varible],
# 													test_size = 0.25,
# 													random_state = 0)

# X_train.info()
# X_train.describe()
# X_train.isnull().sum()
# plot.plot_missing_values(df_train)

index_column = X_test[index_varible]

""" Find missing value coluns"""
ploter = Plot()
missing_value_columns = ploter.get_columns(X_train)

ploter_test = Plot()
missing_value_columns_test = ploter_test.get_columns(X_test)

t_c = [];

""" replace missign categorical data"""
# concat train and test sets to creat the same dummy variables
concateneted_dateset_train = X_train;
concateneted_dateset_test = X_test;

# create dummy variables
for column in  t_c:
	concateneted_dateset_train[column] = concateneted_dateset_train[column].apply(lambda x: 'missing_data' if str(x) == 'nan'  else x);
	dummies_train = pd.get_dummies(concateneted_dateset_train[column], prefix=column, drop_first=True);
	concateneted_dateset_train = pd.concat([concateneted_dateset_train, dummies_train], axis = 1);

	concateneted_dateset_test[column] = concateneted_dateset_test[column].apply(lambda x: 'missing_data' if str(x) == 'nan'  else x);
	dummies_test = pd.get_dummies( concateneted_dateset_test[column], prefix=column, drop_first=True);
	concateneted_dateset_test = pd.concat([ concateneted_dateset_test, dummies_test], axis = 1);

X_train = concateneted_dateset_train.loc[:, concateneted_dateset_train.columns != dependent_varible]
X_test = concateneted_dateset_test.loc[:, concateneted_dateset_test.columns != dependent_varible]

# added new parts untill here

# adding b0 iqual to 1: demanded by the stats library;
b_length = len(x_train);
b_zero = np.ones((b_length, 0), dtype = np.int);
x_train = np.append( arr = b_zero, values = x_train, axis = 1);

# backward elimination
max_p_value = 1;
non_significant_index = -1;
eliminator = None;
while max_p_value > 0.05:
	if not non_significant_index == -1:
		x_train = np.delete(x_train, non_significant_index, 1);
		x_test = np.delete(x_test, non_significant_index, 1);
	sm_result = sm.OLS(endog = y_train, exog = x_train).fit()
	p_values = sm_result.pvalues;
	max_p_value = np.amax(p_values);
	non_significant_index = list(p_values).index(max_p_value);

""" LOGISTIC REGRESSION """
regressor = LogisticRegression(random_state = 0);
regressor.fit(x_train, y_train);
prediction = regressor.predict(x_test);
print(prediction)

# #################
# SUBMIT ANSWER
# #################
holdout_ids = df_test['Id'];
sub_df = {
	"Id":holdout_ids,
	"Cover_Type": prediction	
};

ds = pd.DataFrame(sub_df);
ds.to_csv("sample_submission.csv", index=False);