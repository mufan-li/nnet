import numpy as np
from nnet_functions import *

class nnet(object):
	""" the most basic neural net class
		Containing
			- network structure
			- hyper parameters
			- data points
			- weights
			- relevant values at nodes
	"""

	def __init__(self, xN, hV, yN, bN):
		self.xN = xN # number of inputs # including constant

		if type(hV) != type([1]): # if not list
			hV = [hV]

		self.hV = hV
		self.hN = np.sum(hV) # vector of hidden layers
		self.yN = yN # number of outputs
		self.bN = bN # mini-batch size
		self.N = xN+self.hN+yN
		
		# data
		self.x = np.zeros((self.N,bN))
		self.gx = np.zeros(self.x.shape)
		self.X = np.zeros((xN,bN)) # input
		self.F = np.zeros((yN,bN)) # prediction
		self.Y = np.zeros((yN,bN)) # true value

		# weights
		self.w = w_init(xN,hV,yN)
		self.v = np.zeros(self.w.shape) # momentum
		self.dLdw = np.zeros(self.w.shape)

		# activation function
		self.act = relu;
		self.actd = relud;

	# input data
	def input(self,X,Y):
		nninput(self,X,Y)

	# define fprop/bprop independently
	def fp(self):
		fprop(self)

	def bp(self):
		bprop(self)

if __name__ == "__main__":
	testnn = nnet(2,[2,3],2,5)
	print testnn.w.shape
	print testnn.x.shape
	print testnn.w



