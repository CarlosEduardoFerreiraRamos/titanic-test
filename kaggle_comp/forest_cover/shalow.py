import numpy as np
import pandas as pd
import statsmodels.formula.api as sm;
import matplotlib.pyplot as plt

from plot import Plot as plot
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
# X_train = df_train.loc[:, df_train.columns != dependent_varible]
# y_train = df_train[dependent_varible]
# X_test = df_test

# split the dataset back to train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_train.loc[:, df_train.columns != dependent_varible],
													df_train[dependent_varible],
													test_size = 0.25,
													random_state = 0)

X_train.info()
X_train.describe()
X_train.isnull().sum()
plot.plot_missing_values(df_train)

index_column = X_test[index_varible]

""" Find missing value coluns"""
ploter = Plot()
missing_value_columns = ploter.get_columns(X_train)

ploter_test = Plot()
missing_value_columns_test = ploter_test.get_columns(X_test)

t_c = [];

""" replace missign categorical data"""
# concat train and test sets to creat the same dummy variables
# concateneted_dateset_train = X_train;
# concateneted_dateset_test = X_test;

# # create dummy variables
# for column in  t_c:
# 	concateneted_dateset_train[column] = concateneted_dateset_train[column].apply(lambda x: 'missing_data' if str(x) == 'nan'  else x);
# 	dummies_train = pd.get_dummies(concateneted_dateset_train[column], prefix=column, drop_first=True);
# 	concateneted_dateset_train = pd.concat([concateneted_dateset_train, dummies_train], axis = 1);

# 	concateneted_dateset_test[column] = concateneted_dateset_test[column].apply(lambda x: 'missing_data' if str(x) == 'nan'  else x);
# 	dummies_test = pd.get_dummies( concateneted_dateset_test[column], prefix=column, drop_first=True);
# 	concateneted_dateset_test = pd.concat([ concateneted_dateset_test, dummies_test], axis = 1);

# X_train = concateneted_dateset_train.loc[:, concateneted_dateset_train.columns != dependent_varible]
# X_test = concateneted_dateset_test.loc[:, concateneted_dateset_test.columns != dependent_varible]

"""dealing with numeric missing data"""
# columns_train = X_train.columns[X_train.isnull().any()].tolist();
# imputer_train = Imputer(axis = 0);
# imputer_train.fit(X_train[columns_train]);
# X_train[columns_train] = imputer_train.transform(X_train[columns_train])

# columns_test = X_test.columns[X_test.isnull().any()].tolist();
# imputer_test = Imputer(axis = 0);
# imputer_test.fit(X_test[columns_test]);
# X_test[columns_test] = imputer_test.transform(X_test[columns_test])

"""REMOVE ORIGINAL CATEGORICAL COLUMNS"""
# X_train = X_train.drop(columns = t_c)
# X_test = X_test.drop(columns = t_c)

""" EQUALITY DATE SETS COLUMNS """
# missing_columns = set(X_train.columns) - set(X_test.columns)
# missing_columns_2 = set(X_test.columns) - set(X_train.columns)
# missing_data = pd.DataFrame(0, index=X_test.index, columns=missing_columns)
# missing_data_2 = pd.DataFrame(0, index=X_train.index, columns=missing_columns_2)
# X_test = pd.concat([X_test,missing_data],  axis=1)
# X_train = pd.concat([X_train, missing_data_2], axis=1)

""" FUTURE SCALING THE DATA"""
fitted = StandardScaler().fit(X_train.loc[:, X_train.columns != dependent_varible]);
X_train.loc[:, X_train.columns != dependent_varible] = fitted.transform(X_train.loc[:, X_train.columns != dependent_varible]);
X_test.loc[:, X_test.columns != dependent_varible] = fitted.transform(X_test.loc[:, X_test.columns != dependent_varible]);


"""ADDING b0"""
data_length = len(X_train);
X_train['b0'] = [1]*data_length;

"""BACKWARD ELIMINATION"""
max_p_value = 1;
non_significant_column = None;
eliminator = None;
num = 0
sm_result = None
while max_p_value > 0.05:
	if not non_significant_column == None:
		del X_train[non_significant_column];
		del X_test[non_significant_column];
	sm_result = sm.OLS(endog = y_train, exog = X_train.loc[:, X_train.columns != dependent_varible]).fit();
	sm_result.summary()
	p_values = sm_result.pvalues;
	max_p_value = np.amax(p_values)
	i = np.where(p_values == max_p_value);
	non_significant_column = list(p_values.index[i])[0];

# remove b0
del X_train['b0'];

""" LOGISTIC REGRESSION """
classifier = LogisticRegression(random_state = 0);
classifier.fit(X_train, y_train);
prediction = classifier.predict(X_test);
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