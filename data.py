import numpy as np

class data(object):
	""" basic class for training, validation, and testing data

		contains - x, y, size, (standard deviation, and scaled data)
	"""

	# include ones defaults to True
	def __init__(self, X, Y, incl1 = True):	

		if incl1:
			ones = np.ones((X.shape[0],1))
			X = np.concatenate([ones,X],1)

		self.x = X
		self.y = Y
		self.N = X.shape[0]
		self.xN = X.shape[1]
		self.yN = Y.shape[1]

if __name__ == '__main__':
	test = data(mnist_data['X_test'], mnist_data['Y_test'])

	


