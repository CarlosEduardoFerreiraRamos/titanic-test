import statsmodels.formula.api as sm ;

class Back_Eliminations(object):
	def __init__(self):
		self.regressor_OLS = None;
		pass
		
	def fit_OLS(self, y, X_opt):
		self.regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
		pass

	def sumary(self):
		print(self.regressor_OLS.summary());

		pass