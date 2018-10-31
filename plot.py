import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class Plot(object):
	def __init__(self):
		pass

	@staticmethod
	def plot_missing_values(data_set):
		null_counts = data_set.isnull().sum()/len(data_set)
		plt.figure(figsize=(16,8))
		plt.xticks(np.arange(len(null_counts))+0.5,null_counts.index,rotation='vertical')
		plt.ylabel('fraction of rows with missing data')
		plt.bar(np.arange(len(null_counts)),null_counts)
		plt.show()

	@staticmethod
	def count_missing_values(data_set, non_feature_columns=[]):
		feature_columns = data_set.columns.drop(non_feature_columns)
		iterables = [feature_columns,['count','fraction','seq']]
		index = pd.MultiIndex.from_product(iterables,names=['feature','stat'])
		ids = data_set.id.unique()
		ids.sort()
		return pd.DataFrame(data=None,index=ids,columns=index)

	def plot_set_survived(self, data_set, index, values):
		sex_pivot = data_set.pivot_table(index=index, values=values);
		sex_pivot.plot.bar();
		plt.show();
		pass

	def show_plot(self, data_set):
		data_set.plot.bar();
		plt.show();

	def remove_columns(self, data_frame, columns):
		return data_frame.drop(columns=columns);

	def get_columns(self, data_frame):
		boolean_arrray = data_frame.isnull().any();
		return data_frame.columns[boolean_arrray].tolist();

	def get_columns_index(self, data_frame):
		boolean_arrray = data_frame.isnull().any();
		return data_frame.columns[boolean_arrray];

	def prot_histogram_sruvived(self, set):
		survived = set[set['Survived'] == 1];
		died = set[set['Survived'] == 0];
		survived['Age'].plot.hist(alpha=0.5, color='red', bins=50);
		died['Age'].plot.hist(alpha=0.5, color='blue', bins=50);
		plt.legend(['Survived', 'Died']);
		plt.show();
		pass

	def cut_to_new_column(self,set, cut_points, label_names, column_name, column):
		set[column_name] = pd.cut(set[column], cut_points, labels=label_names);
		return set;

	def cut_survived(self, train, test):
		cut_points = [-1,0,5,12,18,35,60,100];
		label_names = ["Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"]

		train['Age'] = train['Age'].fillna(-0.5);
		train = self.cut_to_new_column(train,cut_points,label_names, 'Age_categories','Age');

		test['Age'] = test['Age'].fillna(-0.5);
		test = self.cut_to_new_column(test,cut_points,label_names, 'Age_categories','Age');

	def plot_cutted(self, train, test):
		self.cut_survived(train, test);

		pivot = train.pivot_table(index="Age_categories",values='Survived')
		pivot.plot.bar()
		plt.show()

	def plot_scatter(self, train, test, predict):
		plt.scatter(X_train, y_train, color = 'red')
		plt.plot(X_train, regressor.predict(X_train), color = 'blue')
		plt.title('Salary vs Experience (Training set)')
		plt.xlabel('Years of Experience')
		plt.ylabel('Salary')
		plt.show()

	def plot_dendogram(self):
		plt.title('Dendogram');
		plt.xlabel('X axis');
		plt.ylabel('Y axis');
		plt.show();

"""
# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

"""
