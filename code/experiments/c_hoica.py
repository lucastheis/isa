#!/usr/bin/env python

"""
Train hierarchical Gaussian-ISA mixtures.
"""

import sys

sys.path.append('./code')

from isa import ISA, GSM
from models import StackedModel, ConcatModel, MoGaussian
from tools import Experiment, preprocess, mapp, logmeanexp, asshmarray
from transforms import LinearTransform, WhiteningTransform, SubspaceGaussianization, Transform
from numpy import *
from numpy.random import rand
from numpy.linalg import slogdet
from glob import glob

mapp.max_processes = 24

Transform.VERBOSITY = 0

first_layers = {
	'8x8': 'results/c_vanhateren/c_vanhateren.7.22062012.075120.xpck',
#	'8x8': 'results/c_vanhateren/c_vanhateren.15.28062012.131634.xpck',
	'16x16': 'results/c_vanhateren/c_vanhateren.10.22072012.124848.xpck'
}


def main(argv):
	seterr(invalid='raise', over='raise', divide='raise')

	experiment = Experiment()

	patch_size = '16x16'
	max_layers = 5
	merge_subspaces = False
	max_iter = 20
	max_data = 100000
	gibbs_num_steps = 0



	### LOAD TEST DATA

	data_test = load('data/vanhateren.{0}.0.npz'.format(patch_size))['data']
	data_test = preprocess(data_test, shuffle=False)



	### LOAD FIRST LAYER

	# load first layer
	model = Experiment(first_layers[patch_size])['model']
	stacked_model = model.model[1]
	isa = stacked_model.model

	# persistent MC samples
	samples = isa.hidden_states()
	if samples.shape[1] < max_data:
		raise RuntimeError("Could not recover persistent states.")
	samples = samples[:, :max_data]

	# reconstruct data points
	data = dot(isa.basis(), samples)

	# drive Markov chain forward
	samples = isa.sample_posterior(data, hidden_states=samples,
		parameters={'gibbs': {'num_iter': gibbs_num_steps}})

	compl_basis = vstack([isa.basis(), isa.nullspace_basis()])
	compl_data = dot(compl_basis, samples)



	### LOAD AND PROCESS AIS SAMPLES

	loglik = []
	ais_samples = []
	ais_weights = []
	indices = []

	# load AIS samples
	for path in glob(first_layers[patch_size][:-4] + 'ais_samples.*.xpck'):
		results = Experiment(path)
		indices = indices + results['indices']
		if isinstance(results['ais_weights'], list):
			results['ais_weights'] = vstack(results['ais_weights'])
		if len(ais_weights):
			ais_weights = hstack([ais_weights, results['ais_weights']])

			for i in range(len(ais_samples)):
				ais_samples[i] = hstack([ais_samples[i], results['samples'][i]])
		else:
			ais_weights = results['ais_weights']
			ais_samples = results['samples']

	print 'AIS weights:', ais_weights.shape
	print len(unique(indices)), 'unique test data points'

	if len(ais_weights):
		loglik.append(loglikelihood(model, data_test[:, indices], ais_weights))

		print
		print '1 layer, {0} [bit/px]'.format(loglik[-1])
		print

		# remove model component from importance weights, since it will be replaced
		for i in range(len(ais_samples)):
			ais_weights[i, :] -= isa.prior_loglikelihood(ais_samples[i]).flatten() - slogdet(compl_basis)[1]

	# map AIS samples to visible space
	for i in range(len(ais_samples)):
		ais_samples[i] = dot(compl_basis, ais_samples[i])


	# these objects will be saved
	experiment['model'] = model
	experiment['loglik'] = loglik



	### MODEL TRAINING

	# move into shared memory
	ais_weights = asshmarray(ais_weights)

	_logjac_dat = zeros([1, data.shape[1]])
	_logjac_ais = zeros([1, ais_samples[0].shape[1]])

	# train remaining layers
	for layer in range(1, max_layers):
		# Gaussianize training data
		sg = SubspaceGaussianization(isa)

		_logjac_dat += sg.logjacobian(compl_data)
		_logjac_ais += sg.logjacobian(ais_samples[0])

		compl_data = sg(compl_data)

		def parfor(i):
			# Gaussianize AIS samples
			ais_weights[i, :] += sg.logjacobian(ais_samples[i]).flatten()
			return sg(ais_samples[i])
		ais_samples = mapp(parfor, range(len(ais_samples)))

		print
		print 'Data:', mean(-sqrt(sum(square(data), 0)) / 2. - data.shape[0] * log(2. * pi) / 2. + _logjac_dat)
		print 'AIS samples:', mean(-sqrt(sum(square(ais_samples[0]), 0)) / 2. - data.shape[0] * log(2. * pi) / 2. + _logjac_ais)
		print

		# train additional layer
		isa = ISA(compl_data.shape[0])
		isa.initialize(compl_data)
		isa.train(compl_data, parameters={
			'verbosity': 1,
			'training_method': 'lbfgs',
			'max_iter': 5,
			'merge_subspaces': False,
			'lbfgs': {
				'max_iter': 50}})
		isa.train(compl_data, parameters={
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
		stacked_model.model = StackedModel(sg, isa)
		stacked_model = stacked_model.model

		# estimate average log-likelihood on test data
		if len(ais_weights):
			ais_weights_ = copy(ais_weights)

			for i in range(len(ais_samples)):
				ais_weights_[i, :] += isa.loglikelihood(ais_samples[i]).flatten()

			loglik.append(loglikelihood(model, data_test[:, indices], ais_weights_))

			print
			print '{0} layer, {1} [bit/px]'.format(layer + 1, loglik[-1])
			print

		# store intermediate results
		try:
			if len(model.model[1].transforms) > 1:
				experiment.save('results/c_hoica/c_hoica.rg.{0}.{1}.xpck'.format(patch_size, layer))
			else:
				experiment.save('results/c_hoica/c_hoica.{0}.{1}.xpck'.format(patch_size, layer))
		except:
			import pdb
			pdb.set_trace()

	return 0



def loglikelihood(model, data, ais_weights):
	"""
	Takes importance weights and computes log-likelihood in pixel space.
	"""

	# transforms
	dct = model.transforms[0]
	wt = model.model[1].transforms[0]

	if len(model.model[1].transforms) > 1:
		rg = model.model[1].transforms[1]
	else:
		rg = None

	loglik_dc = model.model[0].loglikelihood(dct(data)[:1])
	loglik = logmeanexp(ais_weights, 0)
	loglik = loglik_dc + loglik + wt.logjacobian()

	if rg is not None:
		loglik += rg.logjacobian(wt(dct(data)[1:]))

	loglik = loglik / log(2.) / data.shape[0]
	loglik = loglik[:, -isnan(loglik)]

	sem = std(loglik, ddof=1) / sqrt(loglik.size)

	loglik = mean(loglik)

	return loglik



if __name__ == '__main__':
	sys.exit(main(sys.argv))
