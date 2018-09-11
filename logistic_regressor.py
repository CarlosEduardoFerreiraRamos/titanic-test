from sklearn.linear_model import LogisticRegression;

class LogisticRegressor(object):
	def __init__(self):
		self.classifiel = None;

	def build_classifier(self):
		self.classifiel = LogisticRegression(random_state = 0);

	def train_machine(self, x_train, y_train):
		self.classifiel.fit(x_train, y_train);

	def predict(self, x_test):
		return self.classifiel.predict(x_test);		