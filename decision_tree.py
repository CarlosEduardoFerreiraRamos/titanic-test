"""
DECISION TREE

CART -
	 |- REGRESSION TREES
	 |- CLASSIFICATION TREES

study - information entropy

from sklearn.tree import DecisionTreeRegressor;

regressor = DecisionTreeRegressor(random_state=0);
regressor.fit(x_train, y_train);

 # don't need to apply future scaling

"""
from sklearn.tree import DecisionTreeClassifier;

class DecisionTree(object):

	def __init__(self):
		self.classfier = None;

	def create_classifer():
		self.classfier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0);

	def train_machine(x_train, y_train):
		self.classfier.fit(x_train, y_train);

	def predict(x_test):
		return self.classfier.predict(x_test);
