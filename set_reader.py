import pandas as pd

class Set_Reader(object):
	def __init__(self):
		pass

	def read_files(self):
		self.test = pd.read_csv("test.csv");
		self.train = pd.read_csv("train.csv");
		pass

	def read_file(self, file_name):
		file_path = 'data_sets/' + file_name + '.csv';
		return pd.read_csv(file_path);
		
	def print_files_shapes(self):
		print("Dimensions of train: {}".format(self.train.shape))
		print("Dimensions of test: {}".format(self.test.shape))
		pass

	def get_train(self):
		return self.train;

	def get_test(self):
		return self.test;