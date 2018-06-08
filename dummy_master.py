import pandas as pd;

class Dummy_Master(object):
	def __init__(self):
		pass		

	def create_dummy(self, set, column_name, axis):
		dummies = pd.get_dummies(set[column_name], prefix=column_name);
		set = pd.concat([set, dummies], axis=axis);
		return set;

	def generates_dummies(self, list, train, test):
		for column in list:
			train = self.create_dummy(train, column, 1);
			test = self.create_dummy(test, column, 1);