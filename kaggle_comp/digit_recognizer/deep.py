import numpy as np
import pandas as pd
import statsmodels.formula.api as sm;
import matplotlib.pyplot as plt
import keras

from plot import Plot
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

df_train = pd.read_csv('data_sets/digit_rec/train.csv')