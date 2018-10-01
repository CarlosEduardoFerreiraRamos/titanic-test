import matplotlib.pyplot as plt;
from sklearn.linear_model import LogisticRegression;
from main import Main;
from data_set import Data_Set;
from dummy_master import Dummy_Master;
from regressor import Regressor;
from metrics import Metrics;
from back_elimination import Back_Eliminations;
from set_reader import Set_Reader;
from splitter import Splitter;
from plot import Plot;
from process_data import Pre_Process_Data;

import visual-python

m = Main('init');
r = Regressor();
sp = Splitter();
mt = Metrics();
m.print();
be = Back_Eliminations();
pd = Pre_Process_Data();

sr = Set_Reader();
sr.read_files();
# sr.print_files_shapes();
train = sr.get_train();
test = sr.get_test(); 

ploter = Plot();
ploter.cut_survived(train, test);
# ploter.plot_set_survived(sr.get_train(), "Sex", "Survived");
# ploter.plot_set_survived(sr.get_train(), "Pclass" ,"Survived");
# ploter.prot_histogram_sruvived(sr.get_train());
# ploter.cut_survived(sr.get_train(),sr.get_test());

master = Dummy_Master();

arr = ["Pclass","Sex","Age_categories"];
for column in arr:
	train = master.create_dummy(train, column, 1);
	test = master.create_dummy(test, column, 1);

master.generates_dummies(arr, train, test);

# print(train)

# columns = ['Pclass_2', 'Pclass_3', 'Sex_male'];
columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',
       'Age_categories_Missing','Age_categories_Infant',
       'Age_categories_Child', 'Age_categories_Teenager',
       'Age_categories_Young Adult', 'Age_categories_Adult',
       'Age_categories_Senior']

target_column = 'Survived';

r.train_machine(train[columns], train[target_column]);

holdout = test; 

all_X = train[columns]
all_y = train[target_column]

train_x, test_x, train_y, test_y = sp.split(train[columns], train[target_column]);
# toPrint = sr.get_train()['Age'].describe();
# print(toPrint)
r.train_machine(train_x, train_y);
predictions = r.predict(test_x);
accuracy = mt.model_accuracy(test_y, predictions);

regressor_object = Regressor();
reg = regressor_object.get_regressor();

mt.set_cross_score(reg, all_X, all_y, 10)
mt.sort_score();
scores = mt.get_scores();
cross_accurace = mt.get_mean();

regressor_object_1 = Regressor();
regressor_object_1.train_machine(all_X, all_y);
prediction = r.predict(holdout[columns]);

# back_x = train_x;
# be.fit_OLS(all_y, all_X);
# be.sumary();

holdout_ids = holdout["PassengerId"];

sub_df = {
	"PassengerId":holdout_ids,
	"Survived": prediction	
};

ds = Data_Set(sub_df);
ds.to_csv("submission");

print(prediction)
print(holdout_ids)
# print(cross_accurace)

