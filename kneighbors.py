from sklearn.neighbors import KNeighborsClassifier;

class KNeighbors(object):

	def __init__(self):
		self.classifier = None;
		
	# n_neighbors: number of nearest neighbors.
	# metric: 'minkoski' for euclides distancies.
	# p: 2 to euclides distancies;
	def create_calssfier(self):
		self.classifier = KNeighborsClassifier(n_neighbors = 5, metric= 'minkoski', p =2);

	def train_machine(self, x_train, y_train):
		self.classifier.fit(x_train, y_train);

	def predict(self, x_test):
		return self.classifier.predict(x_test);