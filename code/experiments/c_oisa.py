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
	[  '8x8', 1, 1,   20,  10,  True, False,  5, False, False], # 0
	['16x16', 1, 1,   30,  15,  True, False, 10, False, False],

	# overcomplete models
	[  '8x8', 2, 1, 1000, 100,  True, False, 10, False, False], # 2
	['16x16', 2, 1, 1000, 100,  True, False, 10, False, False],

	# overcomplete models with Laplace marginals
	[  '8x8', 2, 1,  200, 100, False, False, 10, False, False], # 4
	[  '8x8', 2, 1,  100, 100, False,  True, 10, False, False],
	['16x16', 2, 1,  200, 100, False,  True, 10, False, False],

	# initialize with sparse coding
	[  '8x8', 2, 1,   50, 200,  True,  True, 10, False, False], # 7
	[  '8x8', 3, 1,   50, 200,  True,  True, 10, False, False],
	[  '8x8', 4, 1,   50, 200,  True,  True, 10, False, False],
	['16x16', 2, 1,   50, 200,  True,  True, 20, False, False],
	['16x16', 3, 1,   50, 200,  True,  True, 20, False, False],
	['16x16', 4, 1,   50, 200,  True,  True, 20, False, False],

	# learning subspaces
	[  '8x8', 1, 1,   30,  20, False, False,  5,  True, False], # 13
	['16x16', 1, 1,   30,  20, False, False, 10,  True, False],

	# with radial Gaussianization
	[  '8x8', 2, 1,   50, 200,  True,  True, 10, False,  True], # 15
	['16x16', 2, 1,   50, 200,  True,  True, 20, False,  True],

	# overcomplete ISA
	[  '8x8', 2, 1,   50, 200,  True,  True, 10,  True, False], # 17
	[  '8x8', 2, 2,   50, 200,  True,  True, 10, False, False],

	# miscellaneous
	[  '8x8', 1, 1,   50, 200,  True,  True, 10, False,  True], # 19
	['16x16', 1, 1,   50, 200,  True,  True, 20, False,  True],
	['16x16', 1, 4,   50, 200,  True,  True, 20, False,  True],
	['16x16', 2, 2,  200, 200,  True, False, 20, False, False],
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

	# discrete cosine transform and whitening transform
	dct = LinearTransform(dim=int(sqrt(data.shape[0])), basis='DCT')
	wt = WhiteningTransform(dct(data)[1:], symmetric=True)



	### MODEL DEFINITION

	if radial_gaussianization:
		isa = ISA(num_visibles=data.shape[0] - 1,
			num_hiddens=data.shape[0] * overcompleteness - 1, ssize=ssize)

		gsm = GSM(data.shape[0] - 1, 20)
		gsm.train(wt(data[1:]), max_iter=100, tol=1e-8)

		rg = RadialGaussianization(gsm)

		# model DC component with a mixture of Gaussians
		model = StackedModel(dct,
			ConcatModel(MoGaussian(20), StackedModel(wt, rg, isa)))

	else:
		isa = ISA(num_visibles=data.shape[0] - 1,
			num_hiddens=data.shape[0] * overcompleteness - 1, ssize=ssize, num_scales=20)

		# model DC component with a mixture of Gaussians
		model = StackedModel(dct,
			ConcatModel(MoGaussian(20), StackedModel(wt, isa)))



	### MODEL TRAINING

	# variables which will be saved
	experiment['model'] = model
	experiment['parameters'] = parameters[int(argv[1])]



	def callback(phase, isa, iteration):
		"""
		Saves intermediate results every few iterations.
		"""

		if any(isnan(isa.subspaces()[0].scales)):
			print 'Scales are NaN.'
			return False

		if not iteration % 5:
			# whitened filters
			A = dot(dct.A[1:].T, isa.A)

			patch_size = int(sqrt(A.shape[0]) + 0.5)

			try:
				# save intermediate results
				experiment.save('results/c_vanhateren.{0}/results.{1}.{2}.xpck'.format(argv[1], phase, iteration))

				# visualize basis
				imsave('results/c_vanhateren.{0}/basis.{1}.{2:0>3}.png'.format(argv[1], phase, iteration),
					stitch(imformat(A.T.reshape(-1, patch_size, patch_size))))
			except:
				print 'Could not save intermediate results.'



	if len(argv) > 2:
		# initialize model with trained model
		results = Experiment(argv[2])
		model = results['model']

		isa = model.model[1].model
		dct = model.transforms[0]
		wt = model.model[1].transforms[0]

		if radial_gaussianization:
			rg = model.model[1].transforms[1]

		experiment['model'] = model

#		if overcompleteness > 1:
#			# reconstruct data without DC component
#			data = dot(isa.A, isa.hidden_states())
#			data = dot(dct.A[1:].T, wt.inverse(rg.inverse(data)))

	else:
		# train mixture of Gaussians on DC component
		model.train(data, 0, max_iter=100)

		# initialize filters and marginals
#		model.initialize(data, 1)
		isa.initialize()

		experiment.progress(10)

		if sparse_coding:
			# initialize with sparse coding
			model.train(data, 1, parameters={
				'training_method': 'mp',
				'mp': {
					'max_iter': max_iter,
					'step_width': 0.01,
					'batch_size': 100,
					'num_coeff': num_coeff},
				'callback': lambda iteration, isa: callback(0, isa, iteration)})
			isa.orthogonalize()

		else:
			# train model using a subset of the data
			model.train(data[:, :20000], 1, parameters={
				'training_method': 'sgd',
				'max_iter': max_iter,
				'persistent': True,
				'train_prior': train_prior,
				'merge_subspaces': False,
				'sgd': {
					'max_iter': 1,
					'batch_size': 100,
					'momentum': 0.8},
				'gibbs': {
					'ini_iter': 5,
					'num_iter': 1
					},
				'callback': lambda iteration, isa: callback(0, isa, iteration)})

	experiment.progress(50)

	# fine-tune model using all the data
	model.train(data, 1, parameters={
		'verbosity': 1,
		'training_method': 'lbfgs',
		'max_iter': max_iter_ft,
		'persistent': True,
		'train_prior': train_prior,
		'merge_subspaces': merge_subspaces,
		'merge': {
			'verbosity': 0,
			'num_merge': model.dim,
			'max_iter': 10,
		},
		'sgd': {
			'max_iter': 1,
			'batch_size': 100,
			'momentum': 0.8},
		'lbfgs': {
			'max_iter': 50},
		'gibbs': {
			'ini_iter': 10 if not len(argv) > 2 and (sparse_coding or not train_prior) else 50,
			'num_iter': 2
			},
		'callback': lambda iteration, isa: callback(1, isa, iteration)})

	experiment.save('results/c_vanhateren/c_vanhateren.{0}.{{0}}.{{1}}.xpck'.format(argv[1]))

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
