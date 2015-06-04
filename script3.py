import numpy as np
import numpy.random as rd
import theano.tensor as T
import theano

from nnet2 import *

# theano.config.exception_verbosity = 'high'
# theano.config.optimizer = 'None'

sN = 100
learning_rate = 1e-1
momentum = 0.9

xval = rd.uniform(0,5,sN).reshape(sN,1)
trainx = theano.shared(
			value = np.matrix(xval,
				dtype = theano.config.floatX),
			borrow = True
			)

yval = np.asarray((xval>2).astype(float),
				dtype = theano.config.floatX)
trainy = theano.shared(
		value = yval,
		borrow = True
		)

x = T.matrix('x')
y = T.matrix('y')
end = T.lscalar()

nn = nnet2(x, xval.shape[1], 2, yval.shape[1], 
	out_act = None)
output = nn.output
cost = nn.mse(y)
gparams = [T.grad(cost, param) for param in nn.params]

vparams = [theano.shared(param.get_value(),borrow=True)
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
	inputs=[end],
	outputs=[cost],
	updates=update1 + update2,
	givens={
		x: trainx[:end],
		y: trainy[:end]
	}
)

bN = sN
for i in range(3):
	print train_model(bN)

# theano.printing.pprint(nn.output)
# theano.printing.debugprint(gparams[0])




