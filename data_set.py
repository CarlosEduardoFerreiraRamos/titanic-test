import pandas as pd;

class Data_Set(object):
	def __init__(self, data_set):
		self.data_set = data_set;
		self.data_frame = self.generate_data_frame();

	def generate_data_frame(self):
		return pd.DataFrame(self.data_set);

	def get_data_frame(self):
		return self.data_frame;

	def to_csv(self, name):
		file_name = name + ".csv";
		self.data_frame.to_csv(file_name, index=False);
		pass