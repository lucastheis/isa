#!/usr/bin/env python

"""
Train ISA model on whitened natural images patches.
"""

import sys

sys.path.append('./code')

from numpy import *
from models import ISA, GSM
from tools import preprocess, Experiment

# PS, SS, OC, ND, MI
parameters = [
	['8x8', 1, 1, 50, 200],
	['8x8', 1, 2, 50, 200],
]

def main(argv):
	if len(argv) < 2:
		print 'Usage:', argv[0], '<params_id>'
		print
		print '  {0:>3} {1:>5} {2:>3} {3:>3} {4:>5} {5:>4}'.format('ID', 'PS', 'SS', 'OC', 'ND', 'MI')

		for id, params in enumerate(parameters):
			print ' {0:>3} {1:>5} {2:>3} {3:>3} {4:>4}k {5:>4}'.format(id, *params)

		print
		print '  ID = parameter set'
		print '  PS = patch size'
		print '  SS = subspace size'
		print '  OC = overcompleteness'
		print '  ND = number of training points'
		print '  MI = maximum number of training epochs'

		return 0

	seterr(invalid='raise', over='raise', divide='raise')

	# some hyperparameters
	patch_size, ssize, overcompleteness, num_data, max_iter = parameters[int(argv[1])]
	num_data *= 1000

	experiment = Experiment()

	# load and whiten data
	data = load('data/vanhateren.{0}.1.npz'.format(patch_size))['data']
	data = preprocess(data)

	# create and train model
	model = ISA(data.shape[0], data.shape[0] * overcompleteness, ssize=ssize)
	model.train(data[:, :num_data])

	# save results
	experiment['model'] = model
	experiment['parameters'] = parameters[int(argv[1])]
	experiment.save('results/experiment01a/experiment01a.{0}.{1}.xpck')

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
