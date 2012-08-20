#!/usr/bin/env python

"""
Train hierarchical Gaussian-ISA mixtures.
"""

import sys

sys.path.append('./code')

from isa import ISA, GSM
from models import StackedModel, ConcatModel, MoGaussian, Mixture
from tools import Experiment, preprocess, mapp
from transforms import LinearTransform, WhiteningTransform, SubspaceGaussianization
from numpy import seterr, asarray, sqrt, load, array
from numpy.random import rand

mapp.max_processes = 1

def main(argv):
	seterr(invalid='raise', over='raise', divide='raise')

	experiment = Experiment()

	patch_size = '8x8'
	max_layers = 20
	merge_subspaces = False
	max_iter = 20
	max_data = 100000



	### DATA PREPROCESSING

	# load data, log-transform and center data
	data = load('data/vanhateren.{0}.2.npz'.format(patch_size))['data']
	data = preprocess(data)

	data_test = load('data/vanhateren.{0}.0.npz'.format(patch_size))['data']
	data_test = preprocess(data_test, shuffle=False)[:, :10000]

	# prepare discrete cosine transform and whitening transform
	dct = LinearTransform(dim=int(sqrt(data.shape[0])), basis='DCT')
	wt = WhiteningTransform(dct(data)[1:], symmetric=True)



	### MODEL TRAINING

	isa = ISA(data.shape[0] - 1)
	stacked_model = StackedModel(wt, isa)
	model = StackedModel(dct, ConcatModel(MoGaussian(20), stacked_model))
	models = [isa]
	loglik = []

	# these objects will be saved
	experiment['model'] = model
	experiment['models'] = models
	experiment['loglik'] = loglik

	# train mixture of Gaussian on DC component
	model.train(data, 0, max_iter=100)

	data_tf = wt(dct(data)[1:])

	# train first layer
	isa.initialize(array(data_tf[:, :max_data]))
	isa.train(array(data_tf[:, :max_data]), parameters={
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
	print

	experiment.save('results/c_hgisa/c_hgisa.0.xpck')

	# train remaining layers
	for layer in range(1, max_layers):
		# Gaussianize data
		sg = SubspaceGaussianization(isa)
		data_tf = sg(data_tf)

		# train additional layer
		isa = ISA(models[-1].num_visibles)
		isa.initialize(array(data_tf[:, :max_data]))
		isa.orthogonalize()
		isa.gaussianity = 0.01
		isa.train(array(data_tf[:, :max_data]), parameters={
			'verbosity': 1,
			'training_method': 'lbfgs',
			'learn_gaussianity': True,
			'max_iter': max_iter,
			'merge_subspaces': merge_subspaces,
			'merge': {
				'verbosity': 0,
				'num_merge': model.dim,
				'max_iter': 10,
			},
			'lbfgs': {
				'max_iter': 50}})

		# explicit representation of mixture distribution
		if isa.gaussianity > 0.:
			mixture = Mixture()
			mixture.components = [GSM(isa.dim, 1), isa]
			mixture.priors = asarray([isa.gaussianity, 1. - isa.gaussianity])
		else:
			mixture = isa

		# replace top layer in model
		if isinstance(stacked_model.model, Mixture):
			stacked_model.model.components[1] = StackedModel(sg, mixture)
			stacked_model = stacked_model.model.components[1]
		else:
			stacked_model.model = StackedModel(sg, mixture)
			stacked_model = stacked_model.model

		# subsample data
		try:
			pw = isa.posterior_weights(data_tf)
			data_tf = data_tf[:, (rand(1, pw.shape[1]) < pw).flatten()]
		except:
			import pdb
			pdb.set_trace()

		# mixture model is now explicitly represented
		isa.gaussianity = 0.

		# store ISA model
		models.append(isa)

		# compute average log-likelihood on test data
		loglik.append(-model.evaluate(data_test))

		print
		print '{0} layers, {1} [bit/px]'.format(layer + 1, loglik[-1])
		print

		# store intermediate results
		try:
			experiment.save('results/c_hgisa/c_hgisa.{0}.{1}.xpck'.format(patch_size, layer))
		except:
			import pdb
			pdb.set_trace()

		if data_tf.shape[1] < 20000:
			print 'Too few data points. Stopping.'
			break
		else:
			print '{0} data points left.'.format(data_tf.shape[1])

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
