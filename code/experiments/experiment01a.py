#!/usr/bin/env python

"""
Train ISA model on whitened natural images patches with noise added.
"""

import sys

sys.path.append('./code')

from numpy import *
from numpy.random import randn
from models import ISA, GSM
from tools import preprocess, Experiment, mapp

mapp.max_processes = 8

from numpy import round
from numpy.linalg import svd

# PS, SS, OC, ND, MI, NS
parameters = [
	['8x8',    1, 1, 100, 150, 32],
	['10x10',  1, 1, 100, 150, 32],
	['12x12',  1, 1, 100, 150, 32],
	['14x14',  1, 1, 100, 150, 32],
	['16x16',  1, 1, 100, 150, 32],
	['8x8',    2, 1, 100, 150, 32],
	['16x16',  2, 1, 100, 150, 32],
	['8x8',   32, 1, 100, 150, 32],
	['16x16', 32, 1, 100, 150, 32],
]

def main(argv):
	if len(argv) < 2:
		print 'Usage:', argv[0], '<params_id>'
		print
		print '  {0:>3} {1:>5} {2:>3} {3:>4} {4:>5} {5:>4} {6:>5}'.format('ID', 'PS', 'SS', 'OC', 'ND', 'MI', 'NS')

		for id, params in enumerate(parameters):
			print '  {0:>3} {1:>5} {2:>3} {3:>3}x {4:>4}k {5:>4} {6:>5}'.format(id, *params)

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

	# load natural image patches
	data = load('data/vanhateren.{0}.1.npz'.format(patch_size))['data']

	# add Gaussian white noise
	if noise_level is not None:
		data += randn(*data.shape) / float(noise_level)

	# log-transform and whiten data
	data, whitening_matrix = preprocess(data, True)

	# create model
	model = ISA(data.shape[0], data.shape[0] * overcompleteness, ssize=ssize)

	# initialize, train and finetune model
	model.train(data[:, :20000], max_iter=40)
	model.train(data[:, :num_data], max_iter=10, method='lbfgs')

	# save results
	experiment['model'] = model
	experiment['parameters'] = parameters[int(argv[1])]
	experiment['whitening_matrix'] = whitening_matrix
	experiment.save('results/experiment01a/experiment01a.{0}.{1}.xpck')

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
