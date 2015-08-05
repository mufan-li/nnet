# new neural network class defined using theano

import numpy as np
import numpy.random as rd
import theano.tensor as T
import theano

from nnet2_functions import *

class nnet_layer(object):
	""" the most basic basic neural net layer class with theano
	"""

	# x - input data
	# n_in - # of nodes in the previous layer
	# n_out - # of nodes in the current layer
	# layer - the layer count so far, for symbol difference
	def __init__(self, x, n_in, n_out, layer=0, act=T.nnet.sigmoid,\
				w = None, b = None):
		if w==None:
			w = theano.shared(
				value=w_init(n_in, n_out),
				name='w'+str(layer),
				borrow=True
			)

		if b==None:
			b = theano.shared(
				value=b_init(n_out),
				name='b'+str(layer),
				borrow=True
			)

		self.w = w
		self.b = b

		lin_output = T.dot(x, self.w) + self.b
		self.output = (
			lin_output if act is None
			else act(lin_output)
		)

		self.params = [self.w, self.b]

class nnet2(object):
	""" the entire neural net implemented with theano
	"""

	def __init__(self, x, n_in, v_hidden, n_out, 
		hid_act = T.nnet.sigmoid, out_act = T.nnet.softmax):

		if type(v_hidden) != type([0]) and \
			type(v_hidden) != type(np.array([])):
			v_hidden = [v_hidden]

		self.n_in = n_in
		self.v_hidden = v_hidden
		self.n_out = n_out

		n_hid = len(v_hidden)
		layers_in = np.concatenate(([n_in],v_hidden))
		layers_out = np.concatenate((v_hidden,[n_out]))
		layers_act = [hid_act] * n_hid + [out_act]

		self.layers = []
		self.params = []
		x_in = x

		for i in range(n_hid+1):
			self.layers.append(
				nnet_layer(x=x_in, n_in=layers_in[i],
					n_out=layers_out[i],
					layer=i,act=layers_act[i])
				)
			self.params += self.layers[i].params
			x_in = self.layers[i].output

		self.output = x_in
		self.outclass = T.argmax(self.output, axis=1)

	def mse(self, y):
		return T.mean(T.square(self.output - y))

	def nll(self, y):
		# return - T.mean(T.dot(T.log(self.output.T), y))
		return -T.mean(
			T.log(self.output)[T.arange(y.shape[0]), 
				T.argmax(y,axis=1)])

	def error(self,y):
		return T.mean(T.neq(self.outclass, T.argmax(y, axis=1)))

if __name__ == "__main__":

	sN = 100
	learning_rate = 1e-5
	momentum = 0.9

	xval = rd.uniform(0,5,sN).reshape(sN,1)
	trainx = theano.shared(
				value = np.matrix(xval,
					dtype = theano.config.floatX),
				borrow = True
				)

	yval = np.asarray((xval>2).astype(float),
					dtype = theano.config.floatX)
	yval = np.concatenate([yval,1-yval],1);
	trainy = theano.shared(
			value = yval,
			borrow = True
			)

	x = T.matrix('x')
	y = T.matrix('y')
	end = T.lscalar()

	nn = nnet2(x, xval.shape[1], 2, yval.shape[1], 
		out_act = T.nnet.softmax)
	output = nn.output
	# cost = nn.mse(y)
	cost = nn.nll(y)
	gparams = [T.grad(cost, param) for param in nn.params]
	
	vparams = [theano.shared(param.get_value(),borrow=True)
				for param in nn.params
	]

	update1 = [
		(vparam, momentum * vparam - learning_rate * gparam) 
		for vparam, gparam in zip(vparams, gparams)
	]
	update2 = [
		(param, param + vparam) 
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
	for i in range(10):
		print train_model(bN)[0]

	# theano.printing.pprint(nn.output)
	# theano.printing.debugprint(gparams[0])




