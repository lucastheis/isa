#!/usr/bin/env python

"""
Train ISA model on whitened natural images patches with noise added.
"""

import sys

sys.path.append('./code')

from numpy import *
from numpy.random import randn
from models import ISA, MoGaussian, StackedModel, ConcatModel, Distribution, GSM
from transforms import LinearTransform, WhiteningTransform, RadialGaussianization
from tools import preprocess, Experiment, mapp

mapp.max_processes = 4
Distribution.VERBOSITY = 2

from numpy import round, sqrt
from numpy.linalg import svd

# PS, SS, OC, ND, MI, NS, MC, LS, RG, TP, MN
parameters = [
	# complete models
	['8x8',     1, 1, 100,  40, 32,  0, False, False, True, False],
	['16x16',   1, 1, 100,  40, 32,  0, False, False, True, False],
	['8x8',     2, 1, 100,  40, 32,  0, False, False, True, False],
	['16x16',   2, 1, 100,  40, 32,  0, False, False, True, False],

	# overcomplete models
	['8x8',     1, 2, 100, 200, 32,  5, False, False, True, False],
	['8x8',     2, 2, 100, 400, 32,  5, False, False, True, False],
	['16x16',   1, 2, 100, 200, 32,  5, False, False, True, False],

	# overcomplete models with noise
	['8x8',     1, 2, 100, 200, 32,  5, False, False, True, True],

	# overcomplete models with Laplace prior
	['8x8',     1, 2, 100, 200, 32,  5, False, False, False, False],
	['16x16',   1, 2, 100, 200, 32,  5, False, False, False, False],

	# special models
	['8x8',     1, 2, 100,  80, 32, 20, False, True,  True, False],
	['8x8',     1, 2, 100,  80, 32, 20, True,  False, True, False],
]

def main(argv):
	if len(argv) < 2:
		print 'Usage:', argv[0], '<params_id> [experiment]'
		print
		print '  {0:>3} {1:>5} {2:>3} {3:>4} {4:>5} {5:>4} {6:>5} {7:>3} {8:>5} {9:>5} {10:>5} {11:>5}'.format(
			'ID', 'PS', 'SS', 'OC', 'ND', 'MI', 'NS', 'MC', 'LS', 'RG', 'TP', 'MN')

		for id, params in enumerate(parameters):
			print '  {0:>3} {1:>5} {2:>3} {3:>3}x {4:>4}k {5:>4} {6:>5} {7:>3} {8:>5} {9:>5} {10:>5} {11:>5}'.format(id, *params)

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
		print '  TP = optimize marginal distributions'
		print '  MN = explicitly model Gaussian noise'
		print
		print '  If an experiment is specified, it will be used to initialize the model parameters.'

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
	radially_gaussianize, \
	train_prior, \
	noise = parameters[int(argv[1])]
	num_data = num_data * 1000

	

	### DATA HANDLING

	# load data, log-transform and center data
	data = load('data/vanhateren.{0}.1.npz'.format(patch_size))['data']
	data = preprocess(data, noise_level=noise_level)
	
	# apply discrete cosine transform
	dct = LinearTransform(dim=int(sqrt(data.shape[0])), basis='DCT')
	data = dct(data)

	# create whitening transform
	wt = WhiteningTransform(data[1:], symmetric=True)

	if noise:
		# noise covariance matrix
		noise = dot(wt.A, wt.A.T) / 20.



	### MODEL DEFINITION

	# create ISA model
	isa = ISA(
		num_visibles=data.shape[0] - 1,
		num_hiddens=(data.shape[0] - 1) * overcompleteness,
		ssize=ssize,
		noise=False)

	if ssize == 1:
		# initialize ISA marginals with Laplace distribution
		isa.initialize(method='laplace')

	if len(argv) > 2:
		# initialize ISA model with already trained model
		results = Experiment(argv[2])

		if results['model'][1].model.num_hiddens != isa.num_hiddens or \
		   results['model'][1].model.num_visibles != isa.num_visibles:
			raise ValueError('Specified model for initialization is incompatible with chosen parameters.')

		isa.A = results['model'][1].model.A
		isa.subspaces = results['model'][1].model.subspaces



	### FURTHER PREPROCESSING

	if radially_gaussianize:
		# radially Gaussianize the data
		gsm = GSM(data.shape[0] - 1, 20)
		gsm.train(wt(data[1:]), max_iter=100, tol=1e-7)

		rg = RadialGaussianization(gsm)

		model = ConcatModel(MoGaussian(10), StackedModel(wt, rg, isa))
	else:
		# model DC component separately with mixture of Gaussians
		model = ConcatModel(MoGaussian(10), StackedModel(wt, isa))



	### TRAIN MODEL

	# train mixture model on DC component
	model.train(data, 0, max_iter=100)

	# turn on a little bit of regularization of the marginals
	for gsm in model[1].model.subspaces:
		gsm.gamma = 1e-3
		gsm.alpha = 2.
		gsm.beta = 1.

	# initialize, train and finetune ISA model
	model.train(data[:, :20000], 1,
		max_iter=100, 
		train_prior=False,
		persistent=True,
		method='sgd', 
		sampling_method=('gibbs', {'num_steps': 2}))

	# save intermediate results
	experiment['parameters'] = parameters[int(argv[1])]
	experiment['transforms'] = [dct, wt]
	experiment['model'] = model
	experiment.save('results/experiment01a/experiment01a.0.{0}.{1}.xpck')

	# train using SGD with regularization turned on
	model.train(data[:, :20000], 1,
		max_iter=max_iter, 
		train_prior=train_prior,
		train_subspaces=train_subspaces,
		init_sampling_steps=10,
		persistent=True,
		method='sgd', 
		sampling_method=('gibbs', {'num_steps': 2}))

	# save intermediate results
	experiment.save('results/experiment01a/experiment01a.1.{0}.{1}.xpck')

	# turn off regularization
	for gsm in model[1].model.subspaces:
		gsm.gamma = 0.

	# turn on additive Gaussian noise
	model.noise = noise

	# train using SGD with regularization turned off
	model.train(data[:, :20000], 1,
		max_iter=100, 
		train_prior=train_prior,
		train_subspaces=train_subspaces,
		init_sampling_steps=10,
		persistent=True,
		method='sgd', 
		sampling_method=('gibbs', {'num_steps': num_steps}))

	# save intermediate results
	experiment.save('results/experiment01a/experiment01a.2.{0}.{1}.xpck')

	if patch_size == '16x16' and overcompleteness > 1:
		# prevent out-of-memory issues by disabling parallelization
		mapp.max_processes = 1

	# train using L-BFGS
	model.train(data[:, :num_data], 1,
		max_iter=20,
		train_prior=train_prior,
		train_subspaces=train_subspaces,
		persistent=True,
		init_sampling_steps=50,
		method='lbfgs', 
		sampling_method=('gibbs', {'num_steps': num_steps}))

	# save results
	experiment.save('results/experiment01a/experiment01a.{0}.{1}.xpck')

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
