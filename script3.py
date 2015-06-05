import numpy as np
import numpy.random as rd
import theano.tensor as T
import theano
import matplotlib.pyplot as plt

from nnet2 import *

# theano.config.exception_verbosity = 'high'
# theano.config.optimizer = 'None'

sN = 100
learning_rate = 1e-1
momentum = 0.5
bN = 10
tN = 1000

xval = rd.uniform(-5,5,sN).reshape(sN,1)
trainx = theano.shared(
			value = np.matrix(xval,
				dtype = theano.config.floatX),
			borrow = True
			)

yval = np.asarray(np.sin(xval),
				dtype = theano.config.floatX)
trainy = theano.shared(
		value = yval,
		borrow = True
		)

x = T.matrix('x')
y = T.matrix('y')
ind = T.lvector()

nn = nnet2(x, xval.shape[1], [10, 10], yval.shape[1], 
	hid_act = T.nnet.sigmoid, out_act = None)
output = nn.output
cost = nn.mse(y)
gparams = [T.grad(cost, param) for param in nn.params]

vparams = [theano.shared(np.zeros(param.get_value().shape),
			borrow=True)
			for param in nn.params
]

update1 = [
	(vparam, momentum * vparam + learning_rate * gparam) 
	for vparam, gparam in zip(vparams, gparams)
]
update2 = [
	(param, param - vparam) 
	for param, vparam in zip(nn.params, vparams)
]

train_model = theano.function(
	inputs=[ind],
	outputs=[cost],
	updates=update1 + update2,
	givens={
		x: trainx[ind],
		y: trainy[ind]
	}
)

predict = theano.function(
	inputs = [ind],
	outputs = [output],
	givens={
		x: trainx[ind]
	}
)


epoch_mse = np.zeros(tN)
for i in range(tN):
	index = rd.randint(0,sN,bN)
	epoch_mse[i] = train_model(index)[0]

pred = np.asarray(predict(range(sN))).reshape(sN,1)

plt.figure()
plt.plot(range(tN),epoch_mse)

plt.figure()
plt.plot(xval,pred,'ro',label='Prediction')
plt.plot(xval,yval,'bo',label='True Value')
plt.show()





