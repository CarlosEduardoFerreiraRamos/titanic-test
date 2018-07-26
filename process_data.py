# library to process the fields with no value;
from sklearn.preprocessing import Imputer;
# library to process categorical data;
from sklearn.preprocessing import LabelEncoder; 
# library to process more than two categorical data;
from sklearn.preprocessing import OneHotEncoder;

from sklearn.preprocessing import StandardScaler;

from sklearn.model_selection import train_test_split;

class Pre_Process_Data(object):
	def __init__(self):
		self.future_scaler = None;
		self.one_hot_encoder = None;
		self.fitted_hot_encoder = None;
		self.label_encoder = None;
		self.fitted_encoded_column = None;
		self.imputer = None;
		self.target_data = None;
		pass

	# the hole processes describe below.
	def replace(self, value, strategy, axis, data):
		self.config_replacer(value, strategy, axis);
		self.replace_missing_data(data);
		return self.return_missing_data(data);

	# Set the imputer with the parameters.
	# value: tells the imputer the value it msut search to replace (ex: NaN); 
	# startegy: tells the imputer how to replace the value (ex: mean);
	# axis: tells the imputer to replace the data using 0 the column or 1 the row; 
	def config_replacer(self, value, strategy, axis):
		axis = 1 if axis =='row' else 0; 
		self.imputer = Imputer(strategy=strategy, missing_values=value, axis=axis);
		pass

	# Replaces the missing data.
	# Isin't necessery to pass al the data set.
	# Can pass only one column (ex: [data[:, index]])
	# or a group of columns (ex: [data[:, indexStartiIncluded : indexEndedNotIncluded])
	# or just a set number of coluns (ex: [data[:, [finstIndx , secoundIndex, ... , lastIndex])
	def replace_missing_data(self, data):
		self.imputer = self.imputer.fit(data);
		pass

	# return the missing data set.
	def return_missing_data(self, data):
		return self.imputer.transform(data);

	# Codes de label column.
	# The especific column mus be pass.
	# In case the column posses more than two labes
	# should be created dumy variables using one hot encoder
	def encode_label_data(self, column):
		self.create_label_encoder();
		self.fit_to_label_encoder(column);
		return self.get_fitted_encoded_column();

	# Creates the LabelEncoder object
	def create_label_encoder(self):
		self.label_encoder = LabelEncoder();
		pass

	# Fit the selected data column to the label encoder
	def fit_to_label_encoder(self, column):
		self.fitted_encoded_column = self.label_encoder.fit_transform(column);

	# return the fitted data;
	def get_fitted_encoded_column(self):
		return self.fitted_encoded_column;

	# The Column that should be coded must be informed by passing the index (ex: [1] or [0,2,3]).
	# Passing the index to the OneHotEncoder constructor will set the categorical features that will be encoded
	# this information will be use to alter the data object.
	def encode_poli_label_data(self, indexs, data):
		self.create_one_hot_encoder(indexs);
		return self.fit_data_to_hot_encoder(data);		

	# Create the OneHotEncoder, and sets the index for the categorical features. 
	def create_one_hot_encoder(self, indexs):
		self.one_hot_encoder = OneHotEncoder(categotical_features = indexs);
		pass

	# Fit the categorical data creating the dumy variabels
	def fit_data_to_hot_encoder(self, all_data):
		return self.one_hot_encoder.fit_transform(all_data).toarray();

	# return X_train,X_test, y_train, y_test;
	# test_size value is a double (ex: 0.2);
	# randon_state will change de value of the splitted set;
	# doing this procedure you don't need to fit the test set after.
	def get_train_test_sets(sefl, X, y, test_size):
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 0);
		return X_train, X_test, y_train, y_test;

	# base: euclides distances
	# Used in analises based in the euclides distance, but not excluding
	# Future scaling transforms the data, to be used more preciselly.
	# It can fit the data do its current model or just transforme it.
	# the data must contain only the data you want to transform/fit (ex; data_train[:, 1:4])
	# CATEGORICAL DATA: those previously transformed to dumy variables can receive a future scale treatment
	# the decision when to apply the future scale must ponder opon if you want to keepit like a refernce or not.
	# if the test set havent been fitted you must fit and transform bouth the train and the test set.      
	def scale_fit_train_test(self, X_train, X_test):
		self.create_future_scaling();
		X_train = self.fit_transform_independents_varibles(X_train);
		X_test = self.fit_transform_independents_varibles(X_test);
		return X_train, X_test; 

	# Fit and transform the Train set  
	def scale_fit_data(self, X_train):
		self.create_future_scaling();
		return self.fit_transform_independents_varibles(X_train);

	# Transform de test set without fitting it to the model!
	def scale_data(self, X_test):
		self.create_future_scaling();
		return self.transform_independents_varibles(X_test);		
	# Creat the StandardScaler obejct.
	def create_future_scaling(self):
		if self.future_scaler == None:
			self.future_scaler = StandardScaler();
			pass

	# Fit and tranform data
	def fit_transform_independents_varibles(self, data):
		return self.future_scaler.fit_transform( data);		

	# Transform data
	def transform_independents_varibles(self, data):
		return self.future_scaler.transform( data);	