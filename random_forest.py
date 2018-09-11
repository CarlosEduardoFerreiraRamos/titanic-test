"""
research: Ensamble Learning


start reandom forest regressor
from sklearn.ensemble omport RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(x_train, y_test)


"""

"""
Ensamble learning: muiltiple machine learning algonrithyms

"""

"""
It's a army of decision trees: every one of them making a prediction, and them entering in consensus.
"""

from sklearn.ensemble import RandomForestClassifier;

class RandomForestClassifier(object):

	def __init__(self):
		self.classfier = None;

	# a problem with incressing the number o n_estimators is taht you can ouverFit your model 
	def create_classifer():
		self.classfier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0);

	def train_machine(x_train, y_train):
		self.classfier.fit(x_train, y_train);

	def predict(x_test):
		return self.classfier.predict(x_test);
	

		