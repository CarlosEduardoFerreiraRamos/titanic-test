import numpy as np;
import pandas as pd;
from plot import Plot;

from sklearn.preprocessing import Imputer;
from sklearn.preprocessing import LabelEncoder; 
from sklearn.preprocessing import OneHotEncoder;


df_train = pd.read_csv('data_sets/titanic/train.csv');
# df_test = pd.read_csv('data_sets/house_prices_test.csv');
passenger_id = df_train['PassengerId']

""" Find missing value coluns"""
ploter = Plot();
missing_value_columns = ploter.get_columns(df_train);

"""dealing with numeric missing data"""
imputer = Imputer(axis=0);
df_train[['Age']] = imputer.fit_transform(df_train[['Age']])

""" replace missign categorical data"""
df_train['Cabin'] = df_train['Cabin'].apply(lambda x: 'missing_data' if str(x) == 'nan'  else x);
df_train['Embarked'] = df_train['Embarked'].apply(lambda x: 'missing_data' if str(x) == 'nan'  else x);

catecorical_array = df_train.dtypes == object
df_train.loc[:, catecorical_array]
df_train.columns[catecorical_array]
# LabelEncoder().fit_transform(df_train.loc[:, ['Survived', 'Sex']])

# concat train and test sets to creat the same dummy variables
concateneted_dateset = df_train;

# categorical features
t_c = ['Survived', 'Sex','Name', 'Ticket', 'Cabin', 'Embarked']

# create dummy variables
for column in  t_c:
	dummies = pd.get_dummies(concateneted_dateset[column], prefix=column);
	concateneted_dateset = pd.concat([concateneted_dateset, dummies], axis = 1);

# split the dataset back to train and test sets
trainIndexs = np.random.rand(len(concateneted_dateset)) < 0.8
df_train = concateneted_dateset[trainIndexs];
df_test = concateneted_dateset[~trainIndexs];
train_ids = df_train["PassengerId"]
test_ids = df_test["PassengerId"]



