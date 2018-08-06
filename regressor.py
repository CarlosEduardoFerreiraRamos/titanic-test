from sklearn.linear_model import LogisticRegression;

class Regressor(object):
	def __init__(self):
		self.lr = LogisticRegression(random_state = 0);
		pass

	def train_machine(self, columns, target_column):
		self.lr.fit(columns, target_column);
		pass

	def predict(self, columns):
		return self.lr.predict(columns);

	def get_regressor(self):
		return self.lr;

