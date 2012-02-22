#!/usr/bin/env python

"""
Train ISA by initializing with parameters of smaller model.
"""

import sys

sys.path.append('./code')

from numpy import *
from numpy.random import randn
from numpy.linalg import inv
from copy import deepcopy
from models import ISA, MoGaussian, StackedModel, ConcatModel, Distribution, GSM
from transforms import LinearTransform, WhiteningTransform, RadialGaussianization
from tools import preprocess, Experiment, mapp

mapp.max_processes = 4
Distribution.VERBOSITY = 2

from numpy import round, sqrt
from numpy.linalg import svd

# PS, SS, OC, ND, MI, NS, MC, LS, RG
parameters = [
	['8x8', 1, 1, 100, 40, 32, 0,  False, False],
]

def main(argv):
	if len(argv) < 2:
		print 'Usage:', argv[0], '<params_id>'
		print
		print '  {0:>3} {1:>5} {2:>3} {3:>4} {4:>5} {5:>4} {6:>5} {7:>3} {8:>5} {9:>5}'.format(
			'ID', 'PS', 'SS', 'OC', 'ND', 'MI', 'NS', 'MC', 'LS', 'RG')

		for id, params in enumerate(parameters):
			print '  {0:>3} {1:>5} {2:>3} {3:>3}x {4:>4}k {5:>4} {6:>5} {7:>3} {8:>5} {9:>5}'.format(id, *params)

		print
		print '  ID = parameter set'
		print '  PS = patch size'
		print '  SS = subspace size'
		print '  OC = overcompleteness'
		print '  ND = number of training points'
		print '  MI = maximum number of training epochs'
		print '  NS = inverse noise level'
		print '  MC = number of Gibbs sampling steps'
		print '  LS = learn subspace sizes'
		print '  RG = radially Gaussianize first'

		return 0

	seterr(invalid='raise', over='raise', divide='raise')



	# start experiment
	experiment = Experiment()



	# hyperparameters
	patch_size, \
	ssize, \
	overcompleteness, \
	num_data, \
	max_iter, \
	noise_level, \
	num_steps, \
	train_subspaces, \
	radially_gaussianize = parameters[int(argv[1])]
	num_data = num_data * 1000




	# load data, log-transform and center data
	data = load('data/vanhateren.8x8.1.npz')['data']
	data = preprocess(data, noise_level=32)

	# initialize ISA model with Laplace marginals
	isa = ISA(data.shape[0], data.shape[0] * overcompleteness, ssize=ssize)
	isa.initialize(method='laplace')

	# initialize, train and finetune ISA model
	isa.train(data[:, :20000],
		max_iter=2,
		train_prior=False,
		method='sgd', 
		sampling_method=('gibbs', {'num_steps': num_steps}))

#	isa.train(data[:, :40000],
#		max_iter=max_iter, 
#		train_prior=True,
#		train_subspaces=train_subspaces,
#		method='sgd', 
#		sampling_method=('gibbs', {'num_steps': num_steps}))
#
#	isa.train(data[:, :num_data_train],
#		max_iter=10,
#		train_prior=True,
#		train_subspaces=train_subspaces,
#		method='lbfgs', 
#		sampling_method=('gibbs', {'num_steps': num_steps}))
#
#

	# turn 8x8 filters into 16x16 filters
	A = isa.A.T.reshape(-1, 8, 8)
	Z = zeros_like(A)
	A = concatenate([
		concatenate([concatenate([A, Z], 1), concatenate([Z, Z], 1)], 2),
		concatenate([concatenate([Z, A], 1), concatenate([Z, Z], 1)], 2),
		concatenate([concatenate([Z, Z], 1), concatenate([A, Z], 1)], 2),
		concatenate([concatenate([Z, Z], 1), concatenate([Z, A], 1)], 2)], 0)
	A = A.reshape(-1, A.shape[0])



	# load data, log-transform and center data
	data = load('data/vanhateren.16x16.preprocessed.npz'.format(patch_size))

	dct = LinearTransform(data['dct'])
	wt = LinearTransform(data['wt'])

	data = data['data_train']

	isa = ISA(data.shape[0], data.shape[0] * overcompleteness, ssize=ssize)
	isa.subspaces = deepcopy(isa.subspaces) + \
		deepcopy(isa.subspaces) + \
		deepcopy(isa.subspaces) + \
		deepcopy(isa.subspaces[:-1])

	isa.A = dot(A, dot(dct.A[1:].T, inv(wt.A)))[1:, :]

	print isa.evaluate(data)

#	# save results
#	experiment['parameters'] = parameters[int(argv[1])]
#	experiment['transforms'] = [dct, wt]
#	experiment['model'] = model
#	experiment.save('results/experiment01a/experiment01a.{0}.{1}.xpck')

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
