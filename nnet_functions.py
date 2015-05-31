import numpy as np

# initialize the weight matrix
def w_init(xN, hV, yN):
	N = xN + np.sum(hV) + yN
	w = np.zeros((N,N))

	layer = [0, xN] + hV + [yN]
	layer = np.cumsum(layer)

	for i in range(len(layer)-2):
		s1 = layer[i]
		e1 = layer[i+1]
		s2 = layer[i+1]
		e2 = layer[i+2]

		w[s1:e1,s2:e2] = np.random.uniform(-0.1,0.1,\
							(e1-s1,e2-s2) )
		# w[s1:e1,s2:e2] = 1;

	return w

# input new data
# - assume new data is not transformed
def nninput(nn,X,Y):
	bN = X.shape[0]
	nn.bN = bN

	nn.x = np.zeros((nn.N,bN))
	nn.gx = np.zeros(nn.x.shape)
	nn.X = X.transpose() # input
	nn.Y = Y.transpose() # output

# relu activation
def relu(x):
	return np.maximum(x,0)
# derivative of relu
def relud(gx):
	return (gx>0).astype(int)

# sigmoid activation
def sigm(x):
	return 1/(1+np.exp(-x))
# derivative of sigmoid
def sigmd(gx):
	return gx * (1-gx)

# forward propagation through a neural net
def fprop(nn):
	# reset from previous fprop
	nn.x *= 0
	nn.gx *= 0
	nn.x[:nn.xN,:] = nn.X
	nn.gx[:nn.xN,:] = nn.X # inputs are not activated

	layer = np.cumsum([0,nn.xN] + nn.hV)
	# hidden layers
	for k in range(len(layer)-2):
		i = np.arange(layer[k],layer[k+1])
		j = np.arange(layer[k+1],layer[k+2])
		nn.x[j,:] = np.dot(nn.w[i][:,j].transpose(), nn.gx[i,:])
		nn.gx[j,:] = nn.act(nn.x[j,:]) # relu activation

	# output layer
	i = j
	j = np.arange(layer[k+2],nn.N)
	nn.x[j,:] = np.dot(nn.w[i][:,j].transpose(), nn.gx[i,:])
	# softmax
	nn.gx[j,:] = np.exp(nn.x[j,:]) / np.sum(np.exp(nn.x[j,:]),0)
	nn.F = nn.gx[j,:]

	return

# backward propagation
def bprop(nn):
	# reset derivatives
	nn.dLdw *= 0
	dLdx = np.zeros(nn.x.shape)
	dgdx = nn.actd(nn.gx) # softmax is ignored here

	layer = np.cumsum([0,nn.xN] + nn.hV)
	# output layer
	n = range(layer[-1],nn.N)
	dgdx[n,:] = 0; # softmax layer
	m = range(layer[-2],layer[-1])
	dLdx[n,:] = nn.Y - nn.gx[n,:]
	nn.dLdw[m[0]:m[-1]+1,n[0]:n[-1]+1] = \
		np.dot(dLdx[n,:], dgdx[m,:].transpose()).transpose()

	# hidden layer
	for k in list(reversed(range(len(layer)-2))):
		l = n; # output indices
		n = m;
		m = range(layer[k],layer[k+1])
		dLdx[n,:] = dgdx[n,:] * np.dot(nn.w[n][:,l], dLdx[l,:])
		nn.dLdw[m[0]:m[-1]+1,n[0]:n[-1]+1] = \
			np.dot(dLdx[n,:], nn.gx[m,:].transpose()).transpose()

	return

# error rate for classification problem
def nnER(nn):
	Find = np.argmax(nn.F,0)
	F1 = np.zeros(nn.F.shape)
	F1[Find,range(nn.bN)] = 1

	# note divide by 2 to avoid double counting
	return np.abs(nn.Y - F1).sum()/nn.bN/2














