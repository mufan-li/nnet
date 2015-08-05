import time
import numpy as np
import numpy.random as rd
import theano.tensor as T
import theano
# import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import pickle

from nnet2 import *

# theano.config.exception_verbosity = 'high'
# theano.config.optimizer = 'None'

print '...loading file'
mnist_file = 'mnist.pkl'
mnist_data = pickle.load( open( mnist_file, "rb" ) )
tmp = np.zeros((mnist_data['y_test'].shape[0],10))

for k in range(mnist_data['y_test'].shape[0]):
	idx = np.mod(mnist_data['y_test'][k],10)
	tmp[k,idx] = 1

mnist_data['Y_test'] = tmp

def shared_dataset(data_x, data_y, borrow=True):
	shared_x = theano.shared(np.matrix(data_x,
									dtype=theano.config.floatX),
							 borrow=borrow)
	shared_y = theano.shared(np.matrix(data_y,
									dtype=theano.config.floatX),
							 borrow=borrow)
	return shared_x, shared_y

train_set_x, train_set_y = shared_dataset(
	mnist_data['X'][:50000,:],mnist_data['Y'][:50000,:])
valid_set_x, valid_set_y = shared_dataset(
	mnist_data['X'][50000:60000,:],mnist_data['Y'][50000:60000,:])
test_set_x, test_set_y = shared_dataset(
	mnist_data['X_test'],mnist_data['Y_test'])

del mnist_data, tmp

sN = train_set_x.get_value().shape[0]
learning_rate = 1e-5
momentum = 0.9
bN = 100
tN = 10 # number of epochs

x = T.matrix('x')
y = T.matrix('y')
ind = T.lvector()

def relu(x):
    return theano.tensor.switch(x<0, 0, x)

nn = nnet2(x, train_set_x.get_value().shape[1], 
		[1000, 1000], train_set_y.get_value().shape[1], 
	hid_act = relu, out_act = T.nnet.softmax)

output = nn.output
outclass = nn.outclass
# cost = nn.mse(y)
cost = nn.nll(y)
cost2 = nn.nll(y)
err = nn.error(y)
gparams = [T.grad(cost, param) for param in nn.params]

vparams = [theano.shared(np.zeros(param.get_value().shape),
			borrow=True)
			for param in nn.params
]

# momentum
update1 = [
	(vparam, momentum * vparam + learning_rate * gparam) 
	for vparam, gparam in zip(vparams, gparams)
]
# change
update2 = [
	(param, param - vparam) 
	for param, vparam in zip(nn.params, vparams)
]

print '...building train function'
train_model = theano.function(
	inputs=[ind],
	outputs=[cost,err],
	updates=update1 + update2,
	givens={
		x: train_set_x[ind],
		y: train_set_y[ind]
	}
)

print '...building validation function'
valid_model = theano.function(
	inputs = [ind],
	outputs = [cost2, err],
	givens = {
		x: valid_set_x[ind],
		y: valid_set_y[ind]
	}
)

print '...building predict function'
predict = theano.function(
	inputs = [ind],
	outputs = [output,outclass],
	givens={
		x: valid_set_x[ind]
	}
)

print '...training'
# epoch_mse = np.zeros(tN)
epoch_nll = np.zeros(tN)
epoch_err = np.zeros(tN)
batchlist = range(sN/bN)

start_time = time.time()

for i in range(tN):

	rd.shuffle(batchlist)
	for j in batchlist:
		index = range(j, j + bN)
		train_model(index)

	# update for each epoch only
	index = rd.randint(0,valid_set_x.get_value().shape[0],10000)
	temp = valid_model(index)
	# epoch_mse[i] = temp[0]
	epoch_nll[i] = temp[0]
	epoch_err[i] = temp[1]

	cur_time = time.time() - start_time
	prog = (i+1.0)/tN
	print 'epoch:', i+1, ', progress:', \
		 np.round(prog*100,1), '%'
	print 'time:', np.round(cur_time,1), 's', \
		', time remaining:', np.round(cur_time/prog*(1-prog)),'s'
	print 'NLL:', np.round(epoch_nll[i],4), \
		', Error:', np.round(epoch_err[i],4), '\n'

# pred = np.asarray(predict(range(sN))).reshape(sN,1)

plt.figure()
plt.plot(range(tN),epoch_nll)

plt.figure()
plt.plot(range(tN),epoch_err)
# plt.figure()
# plt.plot(xval,pred,'ro',label='Prediction')
# plt.plot(xval,yval,'bo',label='True Value')
plt.show()





