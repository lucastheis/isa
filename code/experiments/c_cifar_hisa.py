#!/usr/bin/env python

"""
Train hierarchical ISA.
"""

import sys

sys.path.append('./code')

from isa import ISA, GSM
from models import StackedModel, ConcatModel, MoGaussian, Mixture
from tools import Experiment, preprocess, mapp, cifar
from transforms import LinearTransform, WhiteningTransform, SubspaceGaussianization
from numpy import seterr, asarray, sqrt, load, array, hstack
from numpy.random import rand, permutation

mapp.max_processes = 1

def main(argv):
	seterr(invalid='raise', over='raise', divide='raise')

	experiment = Experiment()

	patch_size = '16x16'
	max_layers = 20
	merge_subspaces = True
	max_iter = 25
	max_data = 200000



	### DATA PREPROCESSING

	# load data, log-transform and center data
	data = cifar.load([1, 2, 3, 4, 5])[0]
	data = cifar.preprocess(data)

	# extract 16x16 patches from CIFAR images and randomize order
	data = hstack(cifar.split(data))
	data = data[:, permutation(data.shape[1])]
	data_test = data[:, max_data:max_data + 20000]
	data_sub = data[:, :20000]
	data = data[:, :max_data]

	# PCA whitening
	wt = WhiteningTransform(data, symmetric=False)
	data = array(wt(data)[1:257], order='F')
	data_test = array(wt(data_test)[1:257], order='F')
	data_sub = array(wt(data_sub)[1:257], order='F')



	### MODEL TRAINING

	isa = ISA(data.shape[0])
	model = StackedModel(isa)
	models = [isa]
	loglik = []

	# these objects will be saved
	experiment['wt'] = wt
	experiment['model'] = model
	experiment['models'] = models
	experiment['loglik'] = loglik

	# train first layer
	isa.initialize(data)
	isa.train(data, parameters={
		'verbosity': 1,
		'training_method': 'sgd',
		'max_iter': max_iter,
		'sgd': {
			'max_iter': 1,
			'batch_size': 100,
			'momentum': 0.8}})
	isa.train(data, parameters={
		'verbosity': 1,
		'training_method': 'lbfgs',
		'max_iter': max_iter,
		'merge_subspaces': merge_subspaces,
		'merge': {
			'verbosity': 0,
			'num_merge': model.dim,
			'max_iter': 10,
		},
		'lbfgs': {
			'max_iter': 50}})

	# compute average log-likelihood on test data
	loglik.append(-model.evaluate(data_test))

	print
	print '1 layer, {0} [bit/px]'.format(loglik[-1])
	print '1 layer, {0} [bit/px] (train)'.format(-model.evaluate(data_sub))
	print

	experiment.save('results/c_cifar_hisa/c_cifar_hisa.{0}.0.xpck'.format(patch_size))

	# train remaining layers
	for layer in range(1, max_layers):
		# Gaussianize data
		sg = SubspaceGaussianization(isa)
		data = array(sg(data), order='F')

		# train additional layer
		isa = ISA(models[-1].num_visibles)
		isa.initialize(data)
		isa.orthogonalize()
		isa.train(data, parameters={
			'verbosity': 1,
			'training_method': 'sgd',
			'max_iter': max_iter,
			'sgd': {
				'max_iter': 1,
				'batch_size': 100,
				'momentum': 0.8}})
		isa.train(data, parameters={
			'verbosity': 1,
			'training_method': 'lbfgs',
			'max_iter': max_iter,
			'merge_subspaces': merge_subspaces,
			'merge': {
				'verbosity': 0,
				'num_merge': model.dim,
				'max_iter': 10,
			},
			'lbfgs': {
				'max_iter': 50}})

		# replace top layer in model
		model.transforms.append(sg)
		model.model = isa

		# store ISA model
		models.append(isa)

		# compute average log-likelihood on test data
		loglik.append(-model.evaluate(data_test))

		print
		print '{0} layers, {1} [bit/px]'.format(layer + 1, loglik[-1])
		print '{0} layers, {1} [bit/px] (train)'.format(layer + 1, -model.evaluate(data_sub))
		print

		# store intermediate results
		try:
			experiment.save('results/c_cifar_hisa/c_cifar_hisa.{0}.{1}.xpck'.format(patch_size, layer))
		except:
			import pdb
			pdb.set_trace()

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
