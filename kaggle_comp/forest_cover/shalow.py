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

# X_train.info()
# X_train.describe()
# X_train.isnull().sum()
# plot.plot_missing_values(df_train)

index_column = X_test[index_varible]

""" Find missing value coluns"""
# ploter = Plot()
# missing_value_columns = ploter.get_columns(X_train)

# ploter_test = Plot()
# missing_value_columns_test = ploter_test.get_columns(X_test)

""" replace missign categorical data"""
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

""" NAIVE BAYES """
from sklearn.naive_bayes import GaussianNB;
classifier = GaussianNB()
classifier.fit(X_train.loc[:, X_train.columns != "Survived"], y_train)
prediction = classifier.predict(X_test.loc[:, X_test.columns != "Survived"])

""" SVM """
from sklearn.svm import SVC
classifier = SVC(C = 1.0 ,kernel = 'rbf', random_state=0)
classifier.fit(X_train.loc[:, X_train.columns != "Survived"], y_train)
prediction = classifier.predict(X_test.loc[:, X_test.columns != "Survived"])

""" KNN """
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric= 'minkowski', p =2)
classifier.fit(X_train.loc[:, X_train.columns != "Survived"], y_train)
prediction = classifier.predict(X_test.loc[:, X_test.columns != "Survived"])

""" DECISON TREE """
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train.loc[:, X_train.columns != "Survived"], y_train)
prediction = classifier.predict(X_test.loc[:, X_test.columns != "Survived"])

""" CONFUSION MATRIX """
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction)

""" CROSS VAL SCORE """
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train.loc[:, X_train.columns != dependent_varible], y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

"""
LOGISTIC REGRESSION							tranforming soil_types to categorical features
[345,  97,   3,   0,  42,   5,  53],		[272, 139,   3,   0,   8,   3, 120], 
[117, 239,  24,   3, 115,  18,   9],		[ 97, 284,  16,   3,  48,   7,  70], 
[  0,   3, 227,  97,  67, 157,   0],		[  0,   9, 163,  99,  65,  41, 174], 
[  0,   0,  23, 498,   0,  23,   0],		[  0,   0,  12, 490,   0,   7,  35], 
[ 12,  66,  56,   0, 353,  37,   1],		[ 18, 131,  48,   0, 173,  40, 115], 
[  1,  23, 105,  60,  54, 316,   0],		[  0,  35,  63,  63,  38,  82, 278], 
[ 60,   1,   0,   0,   0,   0, 470]			[ 23,   2,   0,   0,   0,   0, 506] 
0.6515009961965647							0.6515009961965647
0.00929771844644162							0.00929771844644162

NAYVE BAYS									tranforming soil_types to categorical features
[ 92,   6,   7,   0, 245, 117,  78],		[544,   1,   0,   0,   0,   0,   0],
[ 26,  49,  71,   5, 246, 106,  22],		[518,   7,   0,   0,   0,   0,   0],
[  0,   0, 227, 324,   0,   0,   0],		[517,  34,   0,   0,   0,   0,   0],
[  0,   0,   8, 536,   0,   0,   0],		[341, 203,   0,   0,   0,   0,   0],
[  0,   0, 211,   0, 253,  56,   5],		[525,   0,   0,   0,   0,   0,   0],
[  0,   0, 211, 296,   6,  46,   0],		[554,   5,   0,   0,   0,   0,   0],
[  4,   0,   3,   0,  45,  40, 439]			[517,   1,   0,   0,   0,   0,  13]
0.4343150741221472							0.4343150741221472
0.01144654721813506							0.01144654721813506

SUPORT VECTOR MACHINE						tranforming soil_types to categorical features
[381, 106,   1,   0,  18,   6,  33],		[224, 301,   1,   0,   9,   1,   9],
[105, 299,  21,   1,  77,  13,   9],		[ 72, 370,  18,   1,  55,   6,   3],
[  0,   0, 325,  78,  22, 126,   0],		[  0, 181, 239,  72,  19,  40,   0],
[  0,   0,  11, 517,   0,  16,   0],		[  0,  37,  10, 494,   0,   3,   0],
[ 14,  57,  18,   0, 414,  22,   0],		[ 13, 225,  12,   0, 262,  13,   0],
[  5,  20, 114,  55,  14, 351,   0],		[  7, 305,  69,  48,  14, 116,   0],
[ 62,   1,   0,   0,   0,   0, 468]			[ 52, 324,   0,   0,   0,   0, 155]
0.7303306257383622							0.7303306257383622
0.004673219246577576						0.004673219246577576

KNN											tranforming soil_types to categorical features
[361, 105,   1,   0,  28,   3,  47],		[352, 106,   1,   0,  36,   5,  45],
[124, 283,  23,   1,  71,  14,   9],		[117, 275,  24,   1,  78,  21,   9],
[  0,   4, 358,  60,  19, 110,   0],		[  0,   5, 350,  60,  22, 114,   0],
[  0,   0,  23, 504,   0,  17,   0],		[  0,   0,  23, 505,   0,  16,   0],
[ 15,  30,  20,   0, 444,  16,   0],		[  9,  28,  21,   0, 437,  30,   0],
[  7,   5,  97,  36,  15, 399,   0],		[  7,   4,  93,  38,  15, 402,   0],
[ 35,   5,   0,   0,   3,   0, 488]			[ 41,  14,   0,   0,  11,   3, 462]
0.7523870068194152							0.7523870068194152
0.012815557667136945						0.012815557667136945

DECISON TREE
[359, 133,   3,   0,   9,   3,  38],		[359, 133,   3,   0,   9,   3,  38],
[125, 327,  10,   0,  41,  16,   6],		[125, 325,  10,   0,  41,  18,   6],
[  0,   8, 435,  26,  14,  68,   0],		[  0,   7, 436,  26,  14,  68,   0],
[  0,   0,  15, 508,   0,  21,   0],		[  0,   0,  15, 508,   0,  21,   0],
[ 14,  39,  11,   0, 455,   6,   0],		[ 14,  39,  11,   0, 455,   6,   0],
[  3,   9,  85,  10,  11, 441,   0],		[  3,   9,  86,  10,  11, 440,   0],
[ 37,   5,   0,   0,   0,   0, 489]			[ 41,   5,   0,   0,   0,   0, 485]
0.7898594124406713							0.7898594124406713
0.007735731369329461						0.007735731369329461
"""


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