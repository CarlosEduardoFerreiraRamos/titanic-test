import matplotlib.pyplot as plt;
from main import Main;
from dummy_master import Dummy_Master;
from regressor import Regressor;
from set_reader import Set_Reader;
from plot import Plot;

m = Main('init');
r = Regressor();
m.print();

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

# master.generates_dummies(arr, train, test);

# print(train)

# columns = ['Pclass_2', 'Pclass_3', 'Sex_male'];
columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',
       'Age_categories_Missing','Age_categories_Infant',
       'Age_categories_Child', 'Age_categories_Teenager',
       'Age_categories_Young Adult', 'Age_categories_Adult',
       'Age_categories_Senior']

target_column = 'Survived';

# r.train_machine(train, columns, target_column);

# toPrint = sr.get_train()['Age'].describe();
# print(toPrint)