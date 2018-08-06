"""
result submition:
Id
Cover_Type
 """
from plot import Plot;

from set_reader import Set_Reader;

from process_data import Pre_Process_Data;

from back_elimination import Back_Elimination;

from multiple_linear_regression import Multiple_Linear_Regressor;

from regressor import Regressor;

import numpy as np;

from data_set import Data_Set;

# np.set_printoptions(threshold=np.nan)

""" READ DATA SET FILES """
sr = Set_Reader();
df_train = sr.read_file('forest_cover_types_train');
df_test = sr.read_file('forest_cover_types_test');

""" FINDING MISSNG COLUMNS DATA: in this case none"""
ploter = Plot();
missing_train = ploter.get_columns(df_train);
missing_test = ploter.get_columns(df_test);
print(missing_train);
print(missing_test);

""" DATA FROM DATAFRAME TO ARRAY: sets dosn't posses missing data"""
x_train = df_train.values[:, : -1];
y_train = df_train.values[:, -1];
x_test = df_test.values;

del df_train;
""" ENCODING CATEGORICAL FEATURES: there aren't categorical features with more the two categories"""

""" FUTURE SCALING THE DATA"""
processor_ms = Pre_Process_Data();
x_train,x_test = processor_ms.scale_fit_tranform_train_test(x_train, x_test);
print('precess')

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
	eliminator = Back_Elimination();
	eliminator.fit_OLS(y_train, x_train);
	p_values = eliminator.get_p_values();
	max_p_value = np.amax(p_values);
	non_significant_index = list(p_values).index(max_p_value);
	
""" LOGISTIC REGRESSION """
regressor = Regressor();
regressor.train_machine(x_train, y_train);
prediction = regressor.predict(x_test);
print(prediction)


# #################
# SUBMIT ANSWER
# #################
# print(test.columns);
holdout_ids = df_test['Id'];
sub_df = {
	"Id":holdout_ids,
	"Cover_Type": prediction	
};

ds = Data_Set(sub_df);
ds.to_csv("submission", index=false);
