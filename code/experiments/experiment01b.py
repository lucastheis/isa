#!/usr/bin/env python

"""
Train ISA model on natural images patches with noise added.
"""

import sys

sys.path.append('./code')

from numpy import *
from numpy.random import randn
from models import ISA, GSM
from tools import preprocess, Experiment

# PS, SS, OC, ND, MI, NS
parameters = [
	['8x8',   1, 1, 100, 150, 32.],
	['10x10', 1, 1, 100, 150, 32.],
	['12x12', 1, 1, 100, 150, 32.],
	['14x14', 1, 1, 100, 150, 32.],
	['16x16', 1, 1, 100, 150, 32.],
]

def main(argv):
	if len(argv) < 2:
		print 'Usage:', argv[0], '<params_id>'
		print
		print '  {0:>3} {1:>5} {2:>3} {3:>3} {4:>5} {5:>4} {6:>4}'.format('ID', 'PS', 'SS', 'OC', 'ND', 'MI', 'NS')

		for id, params in enumerate(parameters):
			print '  {0:>3} {1:>5} {2:>3} {3:>3} {4:>4}k {5:>4} {6:>4.0f}'.format(id, *params)

		print
		print '  ID = parameter set'
		print '  PS = patch size'
		print '  SS = subspace size'
		print '  OC = overcompleteness'
		print '  ND = number of training points'
		print '  MI = maximum number of training epochs'
		print '  NS = inverse noise level'

		return 0

	seterr(invalid='raise', over='raise', divide='raise')

	# some hyperparameters
	patch_size, ssize, overcompleteness, num_data, max_iter, noise_level = parameters[int(argv[1])]
	num_data *= 1000

	experiment = Experiment()

	# load and whiten data
	data = load('data/vanhateren.{0}.1.npz'.format(patch_size))['data']

	# log-transform data
	data[data == 0] = 1
	data = log(asarray(data, dtype='float64'))

	# normalize data and add noise
	data -= mean(data)
	data /= std(data)
	data += randn(*data.shape) / float(noise_level)

	# create model
	model = ISA(data.shape[0], data.shape[0] * overcompleteness, ssize=ssize)

	# initialize, train and finetune model
	model.train(data[:, :10000], max_iter=15)
	model.train(data[:, :num_data], max_iter=max_iter)
	model.train(data[:, :num_data], max_iter=5, method=('sgd', {'step_width': 1E-4}))

	# save results
	experiment['model'] = model
	experiment['parameters'] = parameters[int(argv[1])]
	experiment.save('results/experiment01b/experiment01b.{0}.{1}.xpck')

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
