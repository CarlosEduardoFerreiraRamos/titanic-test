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

e: with p-values only:

import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)

with p-values and Adjusted R esqures:

import statsmodels.formula.api as sm
def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
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