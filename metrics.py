from sklearn.metrics import accuracy_score;
from sklearn.model_selection import cross_val_score;


class Metrics(object):
	def __init__(self):
		self.scores: object = None;
		pass

	def model_accuracy(self, target, predictions):
		return accuracy_score(target, predictions);

	def set_cross_score(self, lr, all_X, all_y, cv):
		self.scores = cross_val_score(lr, all_X, all_y, cv=10);

	def sort_score(self):
		self.scores.sort();

	def get_scores(self):
		return self.scores;

	def get_mean(self):
		return self.scores.mean();