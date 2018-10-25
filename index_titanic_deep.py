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
t_c = [ 'Sex', 'Cabin', 'Embarked'] #, 'Ticket','Name'];

# create dummy variables
for column in  t_c:
	dummies = pd.get_dummies(concateneted_dateset[column], prefix=column, drop_first=True);
	concateneted_dateset = pd.concat([concateneted_dateset, dummies], axis = 1);

df_y = concateneted_dateset['Survived']
df_x = concateneted_dateset.loc[:, concateneted_dateset.columns != "Survived"]
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
X_train.columns
import keras
from keras.models import Sequential
from keras.layers import Dense

""" ANN """
classifier = Sequential()
classifier.add(Dense(output_dim = 7, init = 'uniform', activation = 'relu', input_dim = 12))
classifier.add(Dense(output_dim = 7, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# becouse the activation function in the output layer is a sigmoid function prediction will have values raging from 1 to 0  

# add the data
classifier.fit(X_train.loc[:, X_train.columns != "Survived"], y_train, batch_size = 10, nb_epoch = 100)
prediction = classifier.predict(X_test.loc[:, X_test.columns != "Survived"])

# loss
scores = classifier.evaluate(X_train.loc[:, X_train.columns != "Survived"],y_train)

# round prediction
y_pred =  [round(x[0]) for x in prediction]

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

"""
KNN										
[126,  13],								
[ 29,  55]
[0.5684325038316007,
 0.7245508982035929]
"""