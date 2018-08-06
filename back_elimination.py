import statsmodels.formula.api as sm;

"""
 a constant b0 msuts exist, or it must be created
 you can use np to add the columns of onns in the back or, create the column of ones and add the variables in the back.
 *** is require by the statsmodel library
 b_zero = np.append(arr = independent_variables_set, values =  np.ones((50,1)).astype(int))
 or
 b_zero = np.append(arr = np.ones((50,1)).astype(int) , values = independent_variables_set )

 backward elimantion:
 
 *** ordinary least squares
 optimal_array_of_features = 
 regressor = smO.LS(endog = y, enxog = optimal_array_of_features).fit()

 regressor.summary();
 """

class Back_Elimination(object):
	def __init__(self):
		self.regressor_OLS = None;
		self.ols = None;
		pass
		
	def fit_OLS(self, y, X_opt):
		self.regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
		pass

	def get_p_values(self):
		return self.regressor_OLS.pvalues;

	def sumary(self):
		print(self.regressor_OLS.summary());
		pass

	def get_sumary(self):
		return self.regressor_OLS.summary();