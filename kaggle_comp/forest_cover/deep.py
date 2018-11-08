import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import pandas as pd

import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler

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

# X_train.info()
# X_train.describe()
# X_train.isnull().sum()
# plot.plot_missing_values(df_train)
index_column = X_test[index_varible]

""" replace missign categorical data"""
# replace depende varible column

# y_train = pd.get_dummies(y_train, prefix=dependent_varible);
# y_test = pd.get_dummies(y_test, prefix=dependent_varible, drop_first=True);

# concat train and test sets to creat the same dummy variables
concateneted_dateset_train = X_train;
concateneted_dateset_test = X_test;

# create dummy variables
t_c = [column for column in X_train if column.startswith('Soil_Type')];
for column in  t_c:
	concateneted_dateset_train[column] = concateneted_dateset_train[column].apply(lambda x: 'missing_data' if str(x) == 'nan'  else x);
	dummies_train = pd.get_dummies(concateneted_dateset_train[column], prefix=column, drop_first=True);
	concateneted_dateset_train = pd.concat([concateneted_dateset_train, dummies_train], axis = 1);

	concateneted_dateset_test[column] = concateneted_dateset_test[column].apply(lambda x: 'missing_data' if str(x) == 'nan'  else x);
	dummies_test = pd.get_dummies( concateneted_dateset_test[column], prefix=column, drop_first=True);
	concateneted_dateset_test = pd.concat([ concateneted_dateset_test, dummies_test], axis = 1);

X_train = concateneted_dateset_train.loc[:, concateneted_dateset_train.columns != dependent_varible]
X_test = concateneted_dateset_test.loc[:, concateneted_dateset_test.columns != dependent_varible]

"""REMOVE ORIGINAL CATEGORICAL COLUMNS"""
X_train = X_train.drop(columns = t_c)
X_test = X_test.drop(columns = t_c)

""" EQUALITY DATE SETS COLUMNS """
missing_columns = set(X_train.columns) - set(X_test.columns)
missing_columns_2 = set(X_test.columns) - set(X_train.columns)
missing_data = pd.DataFrame(0, index=X_test.index, columns=missing_columns)
missing_data_2 = pd.DataFrame(0, index=X_train.index, columns=missing_columns_2)
X_test = pd.concat([X_test,missing_data],  axis=1)
X_train = pd.concat([X_train, missing_data_2], axis=1)

""" FUTURE SCALING THE DATA"""
fitted = StandardScaler().fit(X_train.loc[:, X_train.columns != dependent_varible]);
X_train.loc[:, X_train.columns != dependent_varible] = fitted.transform(X_train.loc[:, X_train.columns != dependent_varible]);
X_test.loc[:, X_test.columns != dependent_varible] = fitted.transform(X_test.loc[:, X_test.columns != dependent_varible]);
# y_train = fitted.transform(y_train) 

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
y_train = pd.get_dummies(y_train, prefix=dependent_varible);

""" Initialising the ANN"""
classifier = Sequential()
classifier.add(Dense(output_dim = 23, init = 'uniform', activation = 'relu', input_dim = 39))
classifier.add(Dense(output_dim = 23, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 7, init = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

class_pred = classifier.predict_classes(X_test)
# make prediction
y_pred = classifier.predict(X_test)
y_pred_filtered = (y_pred > 0.5)
y_pred_filtered = [round(x[0]) for x in y_pred]
list(x[0] for x in y_pred)
len(class_pred)
y_pred_class = [x + 1 for x in class_pred]

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_class)

from plot import Plot as plot
plot.sum_confusion_matrix(cm)