from skleanr.svm import SVR;

class SVR_Regressor(object):
	def __init__(self):
		
"""
Kernel SVM intuition
when the data is non LINEAR SEPARABLE 

Mapping non LINEAR SEPARABLE set:

	Mapping to higher dimension:

	Kernel trick:

Types of Kernel functions:

	GAUSSIAN RBF Kernel

	Sigmoid Kernel

	Polynomial Kernel
"""


"""
from skleanr.svm import SVR;
# Support Vector Machine (SVM)
# Support Vector Regression (SVR)
# the svr object has the kernel parameter: String, win witch can be set if is linear, poly, rbf(gaussian), sigmoid.
# SVR don't pressents future scaling.

# before runing svr apply FutureScaling.
from standertScaler import standertScaler;
sc_x = StanderScaler()
sc_y = StanderScaler()

x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

regressor = SVR(kernel = 'rbf')
regressor.fit(x_train, y_train);
regressor.predict(x_test);

if you want to predict using standerd scaller you must FutureScale the x_test and them reverser the transformation with
inverse_transform
"""

"""
Kernel SVM
date set must be not linear separable


# Standerd Scaler Must be used: aka Future Scaling

from sklearn.svm import SVC;
classfier = SVC(kernel = 'rbf', random_state = 0);
classfier.fit(x_train, y_train);
y_pred = classfier.predict(x_test)
"""

























