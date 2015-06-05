
# script for training nnet
hV = [1000, 1000]
bN = 100 # mini-batch size
eta0 = 1e-5 # learning rate
tN = int(5e3) # epochs/sgd iterations
fN = 100 # update frequency

nn = nnet(train.xN, hV, train.yN, bN)
er = error(tN)

for t in range(tN):
	bInd = rd.randint(0,train.N,bN)
	# consider creating a method for training
	nn.input(train.x[bInd,:], train.y[bInd,:])

	nn.fp()
	nn.bp()
	eta = eta0 / np.power(2,np.int(t*4/tN)) # reduce learning rate
	nn.v += eta/bN*nn.dLdw
	nn.w += nn.v

	er.trainf(nn,t)

	if (np.mod(t, fN) == 0):
		er.validf(nn,t,valid)
		er.update(t)
		
# add test function for fp
nn.input(test.x[:,:], test.y[:,:])
nn.fp()
print( 'Test MSE: '+str(np.round(
	np.square(nn.Y-nn.F).mean(),4)) )
print( 'Test Error Rate: ' + str(np.round(nn.ER(),4)) )

# plot results
er.plot(nn,test)

np.save('weights', nn.w)
np.save('trainER', er.trainER)
np.save('validER', er.validER)
execfile('email2.py')