"""
Ressions pros and cons        PROS  											CONS

Linear Regression:        Works on any size of dataset, gives information       The linear Regression assuptions
						  about relevance of features.        

Polynomial Regression:    Works on any size of dataset, works very well 		Need to choose the right Polynomial degree
						  on non linear problems 								for a good bias /variance tradeoff

SVR:                      Easy adaptable, works very well on non linear 		Compusory to apply future scaling, not well
						  problems, not biased by outliers. 					know, moredifficult to understand.

Decision Tree Regression: Interpretability, no need for future scaling, 		Poor results on too smalll datasets, 
						  works on bothlinear / not linear problems 			overfitting can easily occur

Random Forest Regerssion: Powerful and accurated, good perfomance 				No interpretability, overfiting can easily
                          on many problems, including not linear 				occur, need to choose the number o trees
"""

"""
Knowing witch regression model use:

First find out if the problem is liner or not linear.

Improving Models:

Prameter tunning. Tunning the hyperparameters like the regularization parameter lambda, or  
the penalty parameter C.
"""

"""
Sliders about Regularization Intuition
https://www.superdatascience.com/wp-content/uploads/2017/02/Regularization.pdf
"""

from sklearn.linear_model import LogisticRegression;

class Regressor(object):

	def __init__(self):
		self.lr = LogisticRegression(random_state = 0);

	def train_machine(self, columns, target_column):
		self.lr.fit(columns, target_column);

	def predict(self, columns):
		return self.lr.predict(columns);

	def get_regressor(self):
		return self.lr;

