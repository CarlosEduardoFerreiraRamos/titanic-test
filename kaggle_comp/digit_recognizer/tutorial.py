# tutorial origin https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6

from plot import Plot as plot 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

train = pd.read_csv("./data_sets/digit_rec/train.csv")
test = pd.read_csv("./data_sets/digit_rec/test.csv")

Y_train = train["label"]
X_train = train.drop(labels = ["label"],axis = 1) # Drop 'label' column
del train # free some space

g = sns.countplot(Y_train)

Y_train.value_counts()
X_train.isnull().any().describe()
# plot.plot_missing_values(Y_train)
test.isnull().any().describe()

# Normalize the data
X_train = X_train / 255.0
test = test / 255.0

# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
# Train and test images (28px x 28px) has been stock into pandas.Dataframe as
# 1D vectors of 784 values. We reshape all data to 28x28x1 3D matrices.
# Keras requires an extra dimension in the end which correspond to channels.
# MNIST images are gray scaled so it use only one channel. For RGB images,
# there is 3 channels, we would have reshaped 784px vectors to 28x28x3 3D matrices.
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
Y_train = to_categorical(Y_train, num_classes = 10)

# Set the random seed
random_seed = 2

# Split the train and the validation set for the fitting
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)

g = plt.imshow(X_train[0][:,:,0])
# plt.show()

"""
The first is the convolutional (Conv2D) layer. It is like a set of learnable filters.
I choosed to set 32 filters for the two firsts conv2D layers and 64 filters for the
two last ones. Each filter transforms a part of the image (defined by the kernel size)
using the kernel filter. The kernel filter matrix is applied on the whole image. Filters
can be seen as a transformation of the image.

The CNN can isolate features that are useful everywhere from these transformed images
(feature maps).

The second important layer in CNN is the pooling (MaxPool2D) layer. This layer simply
acts as a downsampling filter. It looks at the 2 neighboring pixels and picks the maximal
value. These are used to reduce computational cost, and to some extent also reduce
overfitting. We have to choose the pooling size (i.e the area size pooled each time)
more the pooling dimension is high, more the downsampling is important.

Combining convolutional and pooling layers, CNN are able to combine local features and
learn more global features of the image.

Dropout is a regularization method, where a proportion of nodes in the layer are randomly
ignored (setting their wieghts to zero) for each training sample. This drops randomly a
propotion of the network and forces the network to learn features in a distributed way.
This technique also improves generalization and reduces the overfitting.

'relu' is the rectifier (activation function max(0,x). The rectifier activation function
is used to add non linearity to the network.

The Flatten layer is use to convert the final feature maps into a one single 1D vector.
This flattening step is needed so that you can make use of fully connected layers after
some convolutional/maxpool layers. It combines all the found local features of the previous
convolutional layers.

In the end i used the features in two fully-connected (Dense) layers which is just artificial
an neural networks (ANN) classifier. In the last layer(Dense(10,activation="softmax")) the
net outputs distribution of probability of each class.
"""
# Set the CNN model 
# my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out
model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

"""
Once our layers are added to the model, we need to set up a score function,
a loss function and an optimisation algorithm.

We define the loss function to measure how poorly our model performs on
images with known labels. It is the error rate between the oberved labels
and the predicted ones. We use a specific form for categorical classifications
(>2 classes) called the "categorical_crossentropy".

The most important function is the optimizer. This function will iteratively
improve parameters (filters kernel values, weights and bias of neurons ...)
in order to minimise the loss.

I choosed RMSprop (with default values), it is a very effective optimizer. The
RMSProp update adjusts the Adagrad method in a very simple way in an attempt
to reduce its aggressive, monotonically decreasing learning rate. We could also
have used Stochastic Gradient Descent ('sgd') optimizer, but it is slower than
RMSprop.

The metric function "accuracy" is used is to evaluate the performance our model.
This metric function is similar to the loss function, except that the results
from the metric evaluation are not used when training the model (only for evaluation).
"""