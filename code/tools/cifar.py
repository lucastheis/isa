import pickle
from numpy import concatenate, transpose, mean, zeros, asarray

def load(batches=[1], grayscale=False):
	"""
	Load CIFAR batches.
	"""

	data = zeros([32 * 32 * 3, 0])
	labels = []

	for b in batches:
		with open('data/cifar.{0}.pck'.format(b)) as handle:
			pck = pickle.load(handle)

		data = concatenate([data, pck['data'].T], 1)
		labels = concatenate([labels, pck['labels']])

	if grayscale:
		data = transpose(data.T.reshape(-1, 3, 32, 32), [0, 2, 3, 1])
		data = mean(data, 3)
		data = data.reshape(-1, 32 * 32).T

	else:
		data = transpose(data.T.reshape(-1, 3, 32, 32), [0, 2, 3, 1])
		data = data.reshape(-1, 32 * 32 * 3).T

	return asarray(data, order='F'), labels



def preprocess(data, dim=1024):
	"""
	Centers the data.
	"""

	# center data
	data = data - mean(data, 1).reshape(-1, 1)

	return data
