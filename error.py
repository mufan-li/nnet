# error.py
# define class for training/validation errors

import numpy as np
import matplotlib.pyplot as plt
import time

class error(object):
	""" basic class for training, validation, and testing data

		contains - x, y, size, (standard deviation, and scaled data)
	"""

	# include ones defaults to True
	def __init__(self, tN, fN = 100):
		self.tN = tN
		self.fN = fN
		self.train = np.zeros(tN)
		self.valid = np.zeros(tN/fN)
		self.start = time.time()

	def trainf(self,nn,t):
		self.train[t] = np.square(nn.Y-nn.F).mean()

	def validf(self,nn,t,valid):
		nn.input(valid.x,valid.y)
		nn.fp()
		self.valid[t/self.fN] = np.square(nn.Y-nn.F).mean()

	def update(self,t):
		print('Iteration: '+str(t), 
			'Time: '+str(np.round(time.time()-self.start,2))+'seconds',
			'Progress: '+str(np.round(float(t)/self.tN*100))+'%')
		print('Train Error: '+str(np.round(self.train[t],4)),
			'Valid Error: '+str(np.round(self.valid[t/self.fN],4)))
		print('\n')

	def plot(self,nn,valid):
		pltfig = plt.figure(1)

		plt.subplot(211)
		ind = (np.arange(self.tN/self.fN)+1)*self.fN-1
		
		plt.plot(ind,self.train[ind],'r',label='Training')
		plt.plot(ind,self.valid,label='Validation')
		plt.xlabel('Number of Iterations')
		plt.ylabel('Mean Squared Error')
		plt.title('Training & Validation Errors')
		plt.legend()

		plt.subplot(212)
		plt.plot(valid.x[:,1],nn.F[0,:],'ro',label='Prediction')
		plt.plot(valid.x[:,1],valid.y[:,0],'bx',label='True Value')
		plt.title('Prediction vs. True Values')
		plt.legend()

		pltfig.tight_layout()
		plt.show()

if __name__ == '__main__':
	test = error(tN)

	


