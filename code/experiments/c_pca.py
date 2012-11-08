#!/usr/bin/env python

"""
Train overcomplete ICA/ISA on van Hateren image patches.
"""

import sys

sys.path.append('./code')

from models import MoGaussian, StackedModel, ConcatModel, Distribution
from isa import ISA, GSM
from tools import preprocess, Experiment, mapp, imsave, imformat, stitch
from transforms import LinearTransform, WhiteningTransform, RadialGaussianization
from numpy import seterr, sqrt, dot, load, hstack, eye, any, isnan
from numpy.random import rand

# controls parallelization
mapp.max_processes = 1

# controls how much information is printed during training
Distribution.VERBOSITY = 2

# PS, OC, SS, TI, FI, LP, SC, NC, MS, RG
parameters = [
	# complete models
	[  '8x8', 1, 1, 0, 20,  True, False,  5, False, False], # 0
	['16x16', 1, 1, 0, 20,  True, False, 10, False, False], # 1
]

def main(argv):
	if len(argv) < 2:
		print 'Usage:', argv[0], '<param_id>', '[experiment]'
		print
		print '  {0:>3} {1:>7} {2:>5} {3:>5} {4:>5} {5:>5} {6:>5} {7:>5} {8:>5} {9:>5} {10:>5}'.format(
			'ID', 'PS', 'OC', 'SS', 'TI', 'FI', 'LP', 'SC', 'NC', 'MS', 'RG')

		for id, params in enumerate(parameters):
			print '  {0:>3} {1:>7} {2:>5} {3:>5} {4:>5} {5:>5} {6:>5} {7:>5} {8:>5} {9:>5} {10:>5}'.format(id, *params)

		print
		print '  ID = parameter set'
		print '  PS = patch size'
		print '  OC = overcompleteness'
		print '  SS = subspace sizes'
		print '  TI = number of training iterations'
		print '  FI = number of fine-tuning iterations'
		print '  LP = optimize marginal distributions'
		print '  SC = initialize with matching pursuit'
		print '  NC = number of active coefficients in matching pursuit'
		print '  MS = automatically merge subspaces'
		print '  RG = radially Gaussianize first'

		return 0

	seterr(invalid='raise', over='raise', divide='raise')

	# start experiment
	experiment = Experiment()

	# hyperparameters
	patch_size, \
	overcompleteness, \
	ssize, \
	max_iter, \
	max_iter_ft, \
	train_prior, \
	sparse_coding, \
	num_coeff, \
	merge_subspaces, \
	radial_gaussianization = parameters[int(argv[1])]



	### DATA PREPROCESSING

	# load data, log-transform and center data
	data = load('data/vanhateren.{0}.1.npz'.format(patch_size))['data']
	data = data[:, :100000]
	data = preprocess(data, shuffle=False)

	data_test = load('data/vanhateren.{0}.0.npz'.format(patch_size))['data']
	data_test = data_test[:, :10000]
	data_test = preprocess(data_test, shuffle=False)

	# discrete cosine transform and whitening transform
	dct = LinearTransform(dim=int(sqrt(data.shape[0])), basis='DCT')
	wt = WhiteningTransform(dct(data)[1:], symmetric=False)



	### MODEL DEFINITION

	isa = ISA(num_visibles=data.shape[0] - 1,
		num_hiddens=data.shape[0] * overcompleteness - 1, ssize=ssize)

	# model DC component with a mixture of Gaussians
	model = StackedModel(dct,
		ConcatModel(MoGaussian(20), StackedModel(wt, isa)))



	### MODEL TRAINING

	# variables which will be saved
	experiment['model'] = model
	experiment['parameters'] = parameters[int(argv[1])]

	# train mixture of Gaussians on DC component
	model.train(data, 0, max_iter=100)

	# initialize filters and marginals
	model.initialize(data, 1)

	# data is already in PCA basis
	isa.A = eye(isa.dim)

	# fine-tune model using all the data
	model.train(data, 1, parameters={
		'verbosity': 1,
		'training_method': 'lbfgs',
		'max_iter': max_iter_ft,
		'persistent': True,
		'train_basis': False,
		'train_prior': True})

	print '{0:.4f} [bit/px]'.format(model.evaluate(data_test))

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
