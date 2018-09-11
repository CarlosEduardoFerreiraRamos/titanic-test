"""
result submition:
Id
Cover_Type
 """
import numpy as np;

import pandas as pd;

import statsmodels.formula.api as sm;

from sklearn.preprocessing import StandardScaler;

from sklearn.linear_model import LogisticRegression;

""" READ DATA SET FILES """
df_train = pd.read_csv('data_sets/forest_cover_types_train.csv');
df_test = pd.read_csv('data_sets/forest_cover_types_test.csv');

""" DATA FROM DATAFRAME TO ARRAY"""
x_train = df_train.values[:, : -1];
y_train = df_train.values[:, -1];
x_test = df_test.values;

del df_train;

""" FUTURE SCALING THE DATA"""
fs = StandardScaler();
fs.fit(x_train)
x_train = fs.transform(x_train);
x_test = fs.transform(x_test);

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
print(prediction);

# #################
# SUBMIT ANSWER
# #################

holdout_ids = df_test['Id'];
sub_df = {
	"Id":holdout_ids,
	"Cover_Type": prediction	
};

ds = pd.DataFrame(sub_df);
ds.to_csv("submission.csv", index=False);