import pandas as pd
from pandas.io.data import DataReader
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pickle

# my classes
from nnet import *
from data import *

# import data from pickle file
mnist_file = 'mnist.pkl'
mnist_data = pickle.load( open( mnist_file, "rb" ) )
tmp = np.zeros((mnist_data['y_test'].shape[0],10))

for k in range(mnist_data['y_test'].shape[0]):
	idx = np.mod(mnist_data['y_test'][k],10)
	tmp[k,idx] = 1

mnist_data['Y_test'] = tmp

train = data(mnist_data['X'][:50000,:], mnist_data['Y'][:50000,:])
valid = data(mnist_data['X'][50000:,:], mnist_data['Y'][50000:,:])
test = data(mnist_data['X_test'], mnist_data['Y_test'])

## remove large variables
del mnist_data, tmp

# train
# execfile('training.py')
