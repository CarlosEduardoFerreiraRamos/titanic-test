
from sklearn.linear_model import LogisticRegression;

class Regressor(object):
	def __init__(self):
		self.lr = LogisticRegression();
		pass

	def train_machine(self, set, columns, target_column):
		self.lr.fit(set[columns], set[target_column]);
		pass


