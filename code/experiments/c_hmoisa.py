#!/usr/bin/env python

"""
Train hierarchical ISA.
"""

import sys

sys.path.append('./code')

from isa import ISA, GSM
from tools import Experiment, preprocess
from models import MoGaussian, StackedModel, ConcatModel
from transforms import WhiteningTransform, SubspaceGaussianization
from numpy import load

def main(argv):
	experiment = Experiment()

	def callback(phase, iteration, isa):
		"""
		Saves intermediate results every few iterations.
		"""

		if not iteration % 10:
			try:
				# save intermediate results
				experiment.save('results/c_hmoisa/results.{0}.{1}.xpck'.format(phase, iteration))
			except:
				print 'Could not save intermediate results.'

	patch_size = '20x20'
	num_pca = 150
	num_data = 1000000
	max_iter = 200
	overcompleteness = 3

	data = load('data/vanhateren.{0}.1.1.npz'.format(patch_size))['data']
	data = preprocess(data)

	if len(argv) > 1:
		model = Experiment(argv[1])['model']
		experiment['model'] = model
	else:
		model = StackedModel(
			WhiteningTransform(data, symmetric=False),
			ConcatModel(
				MoGaussian(20), 
				ISA(num_pca, ssize=1, num_scales=20), 
				GSM(data.shape[0] - num_pca - 1, 40)))

		experiment['model'] = model

		# train mixture of Gaussians on DC component
		model.train(data[:, :num_data], 0)

		# train GSM on remaining components
		model.train(data[:, :num_data], 2, max_iter=100, tol=1e-8)

		# initialize and train first layer
		model.initialize(data[:, :num_data], 1)
		model.train(data[:, :num_data], 1, parameters={
			'training_method': 'mp',
			'callback': lambda iteration, isa: callback(0, iteration, isa),
			'mp': {
				'max_iter': 50,
				'step_width': 0.01,
				'batch_size': 100,
				'num_coeff': 10}})
		model.train(data[:, :num_data], 1, parameters={
			'verbosity': 1,
			'training_method': 'lbfgs',
			'max_iter': max_iter,
			'callback': lambda iteration, isa: callback(1, iteration, isa),
			'merge_subspaces': True,
			'merge': {
				'verbosity': 0,
				'num_merge': num_pca,
				'max_iter': 10,
			},
			'lbfgs': {
				'max_iter': 100,
				'num_grad': 20}})

	# add second layer
	model.model[1] = StackedModel(
		SubspaceGaussianization(model.model[1]),
		ISA(num_pca, num_pca * overcompleteness, ssize=2, num_scales=20))

	# initialize and train second layer
	model.initialize(data[:, :num_data], 1)
	model.train(data, 1, parameters={
		'training_method': 'mp',
		'callback': lambda iteration, isa: callback(2, iteration, isa),
		'mp': {
			'max_iter': 50,
			'step_width': 0.01,
			'batch_size': 100,
			'num_coeff': 10}})
	model.train(data[:, :num_data], 1, parameters={
		'verbosity': 1,
		'training_method': 'lbfgs',
		'max_iter': max_iter,
		'callback': lambda iteration, isa: callback(3, iteration, isa),
		'persistent': True,
		'gibbs': {
			'ini_iter': 0,
			'num_iter': 2
			},
		'lbfgs': {
			'max_iter': 100,
			'num_grad': 20}})

	experiment.save('results/c_hmoisa/c_hmoisa.{0}.{1}.xpck')

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
