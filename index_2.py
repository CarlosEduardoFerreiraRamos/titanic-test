from set_reader import Set_Reader;
from plot import Plot;
from process_data import Pre_Process_Data;

from regressor import Regressor;

from data_set import Data_Set;

import numpy as np;

import pandas as pd;

# get train and test sets
sr = Set_Reader();
sr.read_files();

"""train should be separeted in train_x and train_y, fist independedt variables and the secound dependent variables"""
train = sr.get_train();
test = sr.get_test();

# remove Name column
ploter = Plot();
train = ploter.remove_columns(train,['Name'])
test = ploter.remove_columns(test,['Name'])

# get missing values columns
missing_value_columns = ploter.get_columns(train);
print(missing_value_columns)

missing_train_value_columns = ploter.get_columns(test)
print(missing_train_value_columns)

# replace missing value fron the columns
processor_ms = Pre_Process_Data();
# replace missing numerical values fron the columns
train[['Age', 'Survived']] = processor_ms.replace('NaN', 'mean', 'columns', train[['Age', 'Survived']]);
test[['Age', 'Pclass']] = processor_ms.replace('NaN', 'mean', 'columns', test[['Age', 'Pclass']]);
test[['Fare', 'Pclass']] = processor_ms.replace('NaN', 'mean', 'columns', test[['Fare', 'Pclass']]);

# replace missing categorical value fron the columns
train['Cabin'] = train['Cabin'].apply(lambda x: 'missing_data' if str(x) == 'nan'  else x);
train['Embarked'] = train['Embarked'].apply(lambda x: 'missing_data' if str(x) == 'nan'  else x);

test['Cabin'] = test['Cabin'].apply(lambda x: 'missing_data' if str(x) == 'nan'  else x);
test['Embarked'] = test['Embarked'].apply(lambda x: 'missing_data' if str(x) == 'nan'  else x);

# Encode binari cotegorical data
train['Sex'] = processor_ms.encode_label_data(train['Sex']);
test['Sex'] = processor_ms.encode_label_data(test['Sex']);

# TODO: right
train_dumy = pd.get_dummies(train['Embarked'], prefix='Embarked', drop_first=True)
train = pd.concat([train, train_dumy], axis=1);
train.drop(['Embarked'],axis=1, inplace=True);

test_dumy = pd.get_dummies(test['Embarked'], prefix='Embarked', drop_first=True)
test = pd.concat([test, test_dumy], axis=1);
test.drop(['Embarked'],axis=1, inplace=True);
test = pd.concat([test, pd.DataFrame({'Embarked_missing_data':[]})], axis=1);
test['Embarked_missing_data'] = test['Embarked_missing_data'].apply(lambda x: 0 if str(x) == 'nan'  else x);

# print(test.columns)
# print(train.columns)

train_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S', 'Embarked_missing_data'];
test_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S', 'Embarked_missing_data'];
# print(train[train_columns].to_string())

train[train_columns],test[test_columns] = processor_ms.scale_fit_train_test(train[train_columns],test[test_columns]);

# print(train[train_columns].to_string())
# print(test[test_columns])

# Regressor
regressor_object_1 = Regressor();
regressor_object_1.train_machine(train[train_columns], train['Survived']);
prediction = regressor_object_1.predict(test[train_columns]);
prediction = prediction.astype(int);
print(prediction);

# #################
# SUBMIT ANSWER
# #################
# print(test.columns);
holdout_ids = test["PassengerId"];
sub_df = {
	"PassengerId":holdout_ids,
	"Survived": prediction	
};

ds = Data_Set(sub_df);
ds.to_csv("normalised_submission_test_fitted");
