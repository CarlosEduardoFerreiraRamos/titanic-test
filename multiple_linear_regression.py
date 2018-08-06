from sklearn.linear_model import LinearRegression;

class Multiple_Linear_Regressor(object):
	def __init__(self):
		self.regressor = None;
		self.prediction = None;
		
	def create_regressor(self):
		self.regressor = LinearRegression();
		pass

	def train_machine(self, x_train, y_train):
		self.regressor.fit(x_train, y_train);

	def predict(self, x_test):
		self.prediction = self.regressor.predict(x_test); 

	def get_prediction(self):
		return self.prediction;  