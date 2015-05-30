# script2.py
# generate sample training data for testing

import pandas as pd
from pandas.io.data import DataReader
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rd

# my classes
from nnet import *
from data import *
from error import *

x = rd.uniform(0,5,1000)
x = np.array(x, ndmin = 2).transpose()
y = np.abs(np.sin(x)) #+ rd.normal(0,0.01,1000)
y = np.concatenate([y, 1-y],1)

train = data(x[:800,:], y[:800,:])
valid = data(x[800:900,:], y[800:900,:])
test = data(x[900:,:], y[900:,:])

del x,y

execfile('training.py')