#!/usr/bin/env python

"""
Train overcomplete ICA/ISA on CIFAR image patches.
"""

import sys

sys.path.append('./code')

from models import MoGaussian, StackedModel, ConcatModel, Distribution
from isa import ISA, GSM
from tools import cifar, preprocess, Experiment, mapp, imsave, imformat, stitch
from transforms import LinearTransform, WhiteningTransform, RadialGaussianization
from numpy import seterr, sqrt, dot, hstack, vstack, eye, zeros, array
from numpy.random import permutation

# controls parallelization
mapp.max_processes = 1

# controls how much information is printed during training
Distribution.VERBOSITY = 2

# PS, OC, SS, TI, FI, LP, SC, NC, MS, RG
parameters = [
	['16x16', 1, 1,   30,  15,  True, False, 10, False, False], # ICA
	['16x16', 2, 1,   50, 200,  True,  True, 20, False, False], # OICA
	['16x16', 1, 1,   30,  20, False, False, 10,  True, False], # ISA
	['16x16', 2, 1,   50, 200,  True,  True, 20, False,  True], # GSM + OICA
	['16x16', 2, 1,   50, 200,  True,  True, 20,  True, False], # OISA
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
	data = cifar.load([1, 2, 3, 4, 5])[0]
	data = cifar.preprocess(data)

	# extract 16x16 patches from CIFAR images and randomize order
	data = hstack(cifar.split(data))
	data = data[:, permutation(data.shape[1])]
	data = data[:, :200000]

	# PCA whitening
	wt = WhiteningTransform(data, symmetric=False)
	data = array(wt(data)[1:257], order='F')



	### MODEL DEFINITION

	if radial_gaussianization:
		gsm = GSM(data.shape[0], 20)
		gsm.train(data, max_iter=100, tol=1e-8)

		rg = RadialGaussianization(gsm)

		# radially Gaussianize data
		data = rg(data)

	isa = ISA(
		num_visibles=data.shape[0],
		num_hiddens=data.shape[0] * overcompleteness,
		ssize=ssize)



	### MODEL TRAINING

	# variables which will be saved
	experiment['wt'] = wt
	experiment['rg'] = rg if radial_gaussianization else None
	experiment['isa'] = isa
	experiment['parameters'] = parameters[int(argv[1])]



	def callback(phase, isa, iteration):
		"""
		Saves intermediate results every few iterations.
		"""

		if not iteration % 5:
			# basis in unwhitened space
			A = isa.A
			A = vstack([
				zeros([1, A.shape[1]]),
				A,
				zeros([wt.dim - 1 - A.shape[0], A.shape[1]])])
			A = wt.inverse(A)
			A = A.T.reshape(-1, 16, 16, 3)

			try:
				# save intermediate results
				experiment.save('results/c_cifar.{0}/results.{1}.{2}.xpck'.format(argv[1], phase, iteration))

				# visualize basis
				imsave('results/c_cifar.{0}/basis.{1}.{2:0>3}.png'.format(argv[1], phase, iteration),
					stitch(imformat(A, perc=99)))
			except:
				print 'Could not save intermediate results.'



	if len(argv) > 2:
		# initialize model with trained model
		results = Experiment(argv[2])
		isa = results['isa']
		wt = results['wt']

		if radial_gaussianization:
			rg = results['rg']

		experiment['wt'] = wt
		experiment['rg'] = rg
		experiment['isa'] = isa

		if overcompleteness > 1:
			# reconstruct data without DC component
			data = dot(isa.A, isa.hidden_states())

	else:
		# initialize filters and marginals
		isa.initialize(data)

		experiment.progress(10)

		if sparse_coding:
			# initialize with sparse coding
			isa.train(data, parameters={
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
			isa.train(data[:, :20000], parameters={
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
	isa.train(data, parameters={
		'verbosity': 1,
		'training_method': 'lbfgs',
		'max_iter': max_iter_ft,
		'persistent': True,
		'train_prior': train_prior,
		'merge_subspaces': merge_subspaces,
		'merge': {
			'verbosity': 0,
			'num_merge': isa.dim,
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

	experiment.save('results/c_cifar/c_cifar.{0}.{{0}}.{{1}}.xpck'.format(argv[1]))

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
