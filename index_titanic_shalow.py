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
passenger_id = df_train['PassengerId']

""" Find missing value coluns"""
ploter = Plot()
missing_value_columns = ploter.get_columns(df_train)

"""dealing with numeric missing data"""
imputer = Imputer(axis=0);
df_train[['Age']] = imputer.fit_transform(df_train[['Age']])

""" replace missign categorical data"""
df_train['Cabin'] = df_train['Cabin'].apply(lambda x: 'missing_data' if str(x) == 'nan'  else x);
df_train['Embarked'] = df_train['Embarked'].apply(lambda x: 'missing_data' if str(x) == 'nan'  else x);

# catecorical_array = df_train.dtypes == object
# df_train.loc[:, catecorical_array]
# df_train.columns[catecorical_array]

del df_train['Name']
del df_train['Ticket']
del df_train["PassengerId"]
# LabelEncoder().fit_transform(df_train.loc[:, ['Survived', 'Sex']])

# concat train and test sets to creat the same dummy variables
concateneted_dateset = df_train;

# categorical features
t_c = [ 'Sex', 'Cabin', 'Embarked'];

# create dummy variables
for column in  t_c:
	dummies = pd.get_dummies(concateneted_dateset[column], prefix=column, drop_first=True);
	concateneted_dateset = pd.concat([concateneted_dateset, dummies], axis = 1);

df_y = df_train['Survived']
df_x = df_train.loc[:, df_train.columns != "Survived"]
# split the dataset back to train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_x,df_y, test_size = 0.25, random_state = 0)

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

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction)

# k-fold cross validation
accuracies = cross_val_score(estimator = classifier, X = X_test.loc[:, X_test.columns != "Survived"], y = y_test, cv = 10)
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
[88, 19],                               [123,  16],                           
[27, 51]                                [ 43,  41]                            
#mean									0.7862146793806317
standert Deviation						0.033766032839746295
"""

"""
Naive Bayes
[104,  35],
[ 31,  53]
0.7212309429700734
0.07138662304387629

SVM (C) unselected						C: 1 & kernel rbf and removin name and tickets	C: 1 & kernel rbf and with name and tickets
[121,  18],								unchenged // 									unchenged //
[ 36,  48]
0.7481272350837569
0.07182253359818928

KNN										n_neighbors: 5 removin name and tickets			n_neighbors: 5 with name and tickets
[115,  24],								unchenged // 									unchenged //
[ 38,  46]
0.7123376623376624
0.0661482248329322
"""
plt.scatter(test_ids, df_test["Survived"], color = 'red')
plt.scatter(test_ids, prediction, color = 'blue')
plt.
plt.title('Linear')
plt.xlabel('Id')
plt.ylabel('Sale Price')
plt.show()
