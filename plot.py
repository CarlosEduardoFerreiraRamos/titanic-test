import matplotlib.pyplot as plt
import pandas as pd

class Plot(object):
	def __init__(self):
		pass

	def plot_set_survived(self, set, index, values):
		sex_pivot = set.pivot_table(index=index, values=values);
		sex_pivot.plot.bar();
		plt.show();
		pass

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

