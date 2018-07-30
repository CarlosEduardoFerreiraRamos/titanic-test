from sklearn.linear_model import LinearRegression;

class Linear_Regression(object):
	"""docstring for Linear_Regression"""
	def __init__(self):
		self.regressor = None;

	def create_regressor(self):
		self.regressor = LinearRegression();

	def fit_to_regressor(self, train_independent, train_dependent):
		self.regressor.fit(train_independent, train_dependent);
		
	def predict(self, test_independent):
		return self.regressor.predict(test_independent);