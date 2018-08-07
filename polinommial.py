from sklearn.linear_model import LinearRegression;
from sklearn.preprocessing import PolynomialFeatures;

# polinomial regression don't need future scaling!

class Polinomial_Regression(object):
	def __init__(self):
		self.lin_reg = None;
		self.poli_reg = None;

		self.create_regressor();  

	def create_regressor(self):
		self.lin_reg = LinearRegression();

	def fit_linear(self, x_train, y_train):
		self.lin_reg.fit(x_train, y_train);

	def create_polinominal_reg(self):
		self.poli_reg = PolynomialFeatures(degree = 2);

	# the contant b0 is added bythe fit_transform method;
	def fit_transform_poly(self, x_train, y_train):
		x_poly = self.poli_reg.fit_tranform(x_train);
		linear_regression = LinearRegression();
		linear_regression.fit(x_poly, y_train);


	"""
	visualising the linear model
	import matplotlib as plt;

	plt.scatter(x_train, y_train, color = 'red')
	plt.plot(x_train, lin_reg.predict(x_train), color = 'blue')
	plt.title('Any title (Linear)')
	plt.xlabel('x axis label')
	plt.ylabel('y axis label')
	plt.show()

	visualising Polynomial
	import matplotlib as plt;

	plt.scatter(x_train, y_train, color = 'red')
	plt.plot(x_train, linear_regresson.predict(self.poli_reg.fit_tranform(x_train)), color = 'blue')
	plt.title('Any title (Polynomial)')
	plt.xlabel('x axis label')
	plt.ylabel('y axis label')
	plt.show()
	"""
