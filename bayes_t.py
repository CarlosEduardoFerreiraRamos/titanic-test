"""
Bayes' Theorem

priorPropability = targetOcurrances / totalOcurrances

marginLikehood =  similarOcurrances / totalOcurrances

Likehood = similarTargetOcurrances / totalTargetOcurrances

posteriorLikehood = (Likehood * priorPropability) / Likehood


Probability of A\\B = Probability of B\\A * (Probability A / Probability B)  
"""

"""
Naíve Bayes

Q: why "Naive"?
Independence of Assunpsions
"""
from sklearn.naive_bays import GaussianNB;

class NaiveBaysClassifier(object):
	def __init__(self):
		self.classfier = None;

	def create_classifier():
		self.classfier = GaussianNB()

	def train_machine(x_train, y_train):
		self.classfier.train(x_train, y_train);

	def predict(x_test):
		return self.classfier.predict(x_test);