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
# df_test = pd.read_csv('data_sets/house_prices_test.csv');

# split the dataset back to train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_train.loc[:, df_train.columns != "Survived"],
													df_train['Survived'],
													test_size = 0.25,
													random_state = 0)

passenger_id = X_test['PassengerId']

""" Find missing value coluns"""
ploter = Plot()
missing_value_columns = ploter.get_columns(X_train)

ploter_test = Plot()
missing_value_columns_test = ploter_test.get_columns(X_test)

"""dealing with numeric missing data"""
imputer = Imputer(axis=0);
X_train[['Age']] = imputer.fit_transform(X_train[['Age']])
X_test[['Age']] = imputer.fit_transform(X_test[['Age']])

""" replace missign categorical data"""
X_train['Cabin'] = X_train['Cabin'].apply(lambda x: 'missing_data' if str(x) == 'nan'  else x);
X_train['Embarked'] = X_train['Embarked'].apply(lambda x: 'missing_data' if str(x) == 'nan'  else x);

X_test['Cabin'] = X_test['Cabin'].apply(lambda x: 'missing_data' if str(x) == 'nan'  else x);

# catecorical_array = df_train.dtypes == object
# df_train.loc[:, catecorical_array]
# df_train.columns[catecorical_array]

del X_train['Name']
del X_train['Ticket']
del X_train['Cabin']
del X_train["PassengerId"]

del X_test['Name']
del X_test['Ticket']
del X_test['Cabin']
del X_test["PassengerId"]
# LabelEncoder().fit_transform(df_train.loc[:, ['Survived', 'Sex']])

# concat train and test sets to creat the same dummy variables
concateneted_dateset_train = X_train;
concateneted_dateset_test = X_test;

# categorical features
t_c = [ 'Sex', 'Embarked'];

# create dummy variables
for column in  t_c:
	dummies_train = pd.get_dummies(concateneted_dateset_train[column], prefix=column, drop_first=True);
	concateneted_dateset_train = pd.concat([concateneted_dateset_train, dummies_train], axis = 1);
	dummies_test = pd.get_dummies( concateneted_dateset_test[column], prefix=column, drop_first=True);
	concateneted_dateset_test = pd.concat([ concateneted_dateset_test, dummies_test], axis = 1);

X_train = concateneted_dateset_train.loc[:, concateneted_dateset_train.columns != "Survived"]
X_test = concateneted_dateset_test.loc[:, concateneted_dateset_test.columns != "Survived"]

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
	sm_result.summary()
	max_p_value = np.amax(p_values)
	i = np.where(p_values == max_p_value);
	non_significant_column = list(p_values.index[i])[0];

# remove b0
del X_train['b0'];

""" LOGISTIC REGRESSION """
classifier = LogisticRegression(random_state=42)
classifier.fit(X_train.loc[:, X_train.columns != "Survived"], y_train)
prediction = classifier.predict(X_test.loc[:, X_test.columns != "Survived"])

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

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction)

# k-fold cross validation
accuracies = cross_val_score(estimator = classifier, X = X_train.loc[:, X_train.columns != "Survived"], y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

# grid search 
parameters = [
	{'n_neighbors': [4,5,6]}
]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

grid_search = grid_search.fit(X_train.loc[:, X_train.columns != "Survived"], y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

"""
LogisticRegression
without Droping Dummy first varible     dropping dummy first varible        the later and using the tickets columns
[88, 19],                               [116,  23],                           
[27, 51]                                [ 23,  61]                            
#mean									0.798541313758705
standert Deviation						0.07817332352866786
"""

"""
Naive Bayes
[104,  35],
[ 31,  53]
0.7212309429700734
0.07138662304387629

SVM (C) unselected						C: 1 & kernel rbf and removin name and tickets	C: 1 & kernel rbf and with name and tickets
[118,  21],								unchenged // 									unchenged //
[ 26,  58]
0.7838791643139469
0.05547009091778175

KNN (with cabin)			KNN (Without Cabin)
[128,  11],					[125, 14],
[ 30,  54]					[ 25, 59]
0.8084803256445048			0.7950474898236092
0.03474507800554288			0.03719964696413435

DECISION TREES
[117,  22],
[ 30,  54]
0.7358742706568794
0.06814500839632841
"""
plt.scatter(test_ids, df_test["Survived"], color = 'red')
plt.scatter(test_ids, prediction, color = 'blue')
plt.
plt.title('Linear')
plt.xlabel('Id')
plt.ylabel('Sale Price')
plt.show()
