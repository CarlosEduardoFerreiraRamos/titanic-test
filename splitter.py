from sklearn.model_selection import train_test_split;

class Splitter(object):
	def __init__(self):
		pass

	def split(self, columns, target_column):
		return train_test_split(columns, target_column, test_size=0.2, random_state=0);
		