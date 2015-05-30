
# script for training nnet
hV = [100, 100]
bN = 50 # mini-batch size
eta = 1e-5 # learning rate
tN = int(1e3) # epochs/sgd iterations
fN = 100 # update frequency

nn = nnet(train.xN, hV, train.yN, bN)
er = error(tN)

for t in range(tN):
	# insert function for selecting mini-batch
	nn.input(train.x[:,:], train.y[:,:])

	nn.fp()
	nn.bp()
	nn.v += eta/bN*nn.dLdw
	nn.w += nn.v

	er.trainf(nn,t)

	if (np.mod(t, fN) == 0):
		er.validf(nn,t,valid)
		er.update(t)
		
# add test function for fp
nn.input(test.x[:,:], test.y[:,:])
nn.fp()
print( 'Test Error: '+str(np.round(
	np.square(nn.Y-nn.F).mean()*100,2)) + '%' )

# plot results
er.plot(nn,valid)