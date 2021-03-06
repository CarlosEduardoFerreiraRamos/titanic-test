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


df_train = pd.read_csv('data_sets/titanic/train.csv')
df_test = pd.read_csv('data_sets/titanic/test.csv');
passenger_id_train = df_train['PassengerId']
passenger_id_test = df_test['PassengerId']

""" Find missing value coluns"""
ploter = Plot()
missing_value_columns = ploter.get_columns(df_train)

ploter_test = Plot()
missing_value_columns_test = ploter_test.get_columns(df_test)

"""dealing with numeric missing data"""
imputer = Imputer(axis=0);
df_train[['Age']] = imputer.fit_transform(df_train[['Age']])
df_test[['Age']] = imputer.fit_transform(df_test[['Age']])
df_test[['Fare']] = imputer.fit_transform(df_test[['Fare']])

""" replace missign categorical data"""
df_train['Cabin'] = df_train['Cabin'].apply(lambda x: 'missing_data' if str(x) == 'nan'  else x);
df_train['Embarked'] = df_train['Embarked'].apply(lambda x: 'missing_data' if str(x) == 'nan'  else x);

df_test['Cabin'] = df_test['Cabin'].apply(lambda x: 'missing_data' if str(x) == 'nan'  else x);
df_test['Embarked'] = df_test['Embarked'].apply(lambda x: 'missing_data' if str(x) == 'nan'  else x);

# catecorical_array = df_train.dtypes == object
# df_train.loc[:, catecorical_array]
# df_train.columns[catecorical_array]

del df_train['Name']
del df_train['Ticket']
del df_train["PassengerId"]

del df_test['Name']
del df_test['Ticket']
del df_test["PassengerId"]
# LabelEncoder().fit_transform(df_train.loc[:, ['Survived', 'Sex']])

# concat train and test sets to creat the same dummy variables
concateneted_dateset_train = df_train;
concateneted_dateset_test = df_test;

# categorical features
t_c = [ 'Sex', 'Cabin', 'Embarked'];

# create dummy variables
for column in  t_c:
	dummies_train = pd.get_dummies(concateneted_dateset_train[column], prefix=column, drop_first=True);
	concateneted_dateset_train = pd.concat([concateneted_dateset_train, dummies_train], axis = 1);
	dummies_test = pd.get_dummies( concateneted_dateset_test[column], prefix=column, drop_first=True);
	concateneted_dateset_test = pd.concat([ concateneted_dateset_test, dummies_test], axis = 1);

y_train = concateneted_dateset_train['Survived']
X_train = concateneted_dateset_train.loc[:, concateneted_dateset_train.columns != "Survived"]

# y_test = concateneted_dateset_test['Survived']
X_test = concateneted_dateset_test.loc[:, concateneted_dateset_test.columns != "Survived"]
# split the dataset back to train and test sets
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(df_x,df_y, test_size = 0.25, random_state = 0)

""" EGUALATE DATE SETS COLUMNS """
missing_columns = set(X_train.columns) - set(X_test.columns)
missing_columns_2 = set(X_test.columns) - set(X_train.columns)
missing_data = pd.DataFrame(0, index=np.arange(len(X_test)), columns=missing_columns)
missing_data_2 = pd.DataFrame(0, index=np.arange(len(X_train)), columns=missing_columns_2)
X_test = pd.concat([X_test,missing_data],  axis=1)
X_train = pd.concat([X_train, missing_data_2], axis=1)

# trainIndexs = np.random.rand(len(concateneted_dateset)) < 0.8
# df_train = concateneted_dateset[trainIndexs]
# df_test = concateneted_dateset[~trainIndexs]
# train_ids = df_train["PassengerId"]
# test_ids = df_test["PassengerId"]
# del df_train["PassengerId"]
# del df_test["PassengerId"]

"""REMOVE ORIGINAL CATEGORICAL COLUMNS"""
X_train = X_train.drop(columns = t_c)
X_test = X_test.drop(columns = t_c)

""" FUTURE SCALING THE DATA"""
fitted = StandardScaler().fit(X_train.loc[:, X_train.columns != "Survived"]);
X_train.loc[:, X_train.columns != "Survived"] = fitted.transform(X_train.loc[:, X_train.columns != "Survived"]);
X_test.loc[:, X_test.columns != "Survived"] = fitted.transform(X_test.loc[:, X_test.columns != "Survived"]);

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
	sm_result = sm.OLS(endog = y_train, exog = X_train.loc[:, X_train.columns != "Survived"]).fit();
	p_values = sm_result.pvalues;
	max_p_value = np.amax(p_values)
	i = np.where(p_values == max_p_value);
	non_significant_column = list(p_values.index[i])[0];

# remove b0
del X_train['b0'];

""" KNN """
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric= 'minkowski', p =2)
classifier.fit(X_train.loc[:, X_train.columns != "Survived"], y_train)
prediction = classifier.predict(X_test.loc[:, X_test.columns != "Survived"])


holdout_ids = passenger_id_test;
sub_df = {
	"PassengerId":holdout_ids,
	"Survived": prediction	
};

ds = pd.DataFrame(sub_df);
ds.to_csv("predict_data/titanic/submission.csv", index=False);

