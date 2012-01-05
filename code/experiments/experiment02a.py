"""
Train hierarchical ISA.
"""

import sys

sys.path.append('./code')

from numpy import *
from models import ISA, Distribution
from transforms import SubspaceGaussianization
from tools import Experiment, preprocess, mapp

Distribution.VERBOSITY = 1

# number of processors used
mapp.max_processes = 8

# PS, SS, ND, MI, NS, ML
parameters = [
	['8x8',   1, 100, 40, 32, 50],
	['16x16', 1, 100, 80, 32, 50],
	['8x8',   2, 100, 40, 32, 50],
	['16x16', 2, 100, 80, 32, 50],
]

def main(argv):
	if len(argv) < 2:
		print 'Usage:', argv[0], '<params_id>'
		print
		print '  {0:>3} {1:>5} {2:>3} {3:>5} {4:>4} {5:>5} {6:>3}'.format('ID', 'PS', 'SS', 'ND', 'MI', 'NS', 'ML')

		for id, params in enumerate(parameters):
			print '  {0:>3} {1:>5} {2:>3} {3:>4}k {4:>4} {5:>5} {6:>3}'.format(id, *params)

		print
		print '  ID = parameter set'
		print '  PS = patch size'
		print '  SS = subspace size'
		print '  ND = number of training points'
		print '  MI = maximum number of training epochs'
		print '  NS = inverse noise level'
		print '  ML = maximum number of layers'

		return 0

	seterr(invalid='raise', over='raise', divide='raise')

	# hyperparameters
	patch_size, ssize, num_data, max_iter, noise_level, max_layers = parameters[int(argv[1])]
	num_data *= 1000

	# start experiment
	experiment = Experiment()

	# load natural image patches
	data = load('data/vanhateren.{0}.1.npz'.format(patch_size))['data']

	# log-transform data, whiten data, and add some noise
	data, whitening_matrix = preprocess(data, return_whitening_matrix=True, noise_level=noise_level)

	# container for hierarchical model
	model = []

	# average log-Jacobian determinant
	logjacobian = 0.

	experiment['parameters'] = parameters[int(argv[1])]
	experiment['whitening_matrix'] = whitening_matrix
	experiment['model'] = model

	for _ in range(max_layers - 1):
		model.append(ISA(data.shape[0], data.shape[0], ssize=ssize))
		
		# initialize, train and finetune model
		model[-1].train(data[:, :20000], max_iter=20, method=('sgd', {'max_iter': 1}), train_prior=False)
		model[-1].train(data[:, :20000], max_iter=max_iter, method=('sgd', {'max_iter': 1}))
		model[-1].train(data[:, :num_data], max_iter=10, method='lbfgs')

		# save model
		experiment['num_layers'] = len(model)
		experiment.save('results/experiment02a/experiment02a.{0}.{1}.xpck')

		# evaluate hierarchical model on training data
		loglik = mean(model[-1].loglikelihood(data[:, :num_data])) + logjacobian
		print
		print '{0:>2}, {1} [bit/pixel]'.format(len(model), -loglik / data.shape[0] / log(2.))
		print

		# subspace Gaussianization transform
		sg = SubspaceGaussianization(model[-1])
		logjacobian += mean(sg.logjacobian(data[:, :num_data]))
		data = sg(data[:, :num_data])

	# top layer
	model.append(ISA(data.shape[0], data.shape[0], ssize=ssize))
	
	# train and finetune model
	model[-1].train(data[:, :20000], max_iter=max_iter, method=('sgd', {'max_iter': 1}))
	model[-1].train(data[:, :num_data], max_iter=10, method='lbfgs')

	# save model
	experiment['num_layers'] = max_layers
	experiment.save('results/experiment02a/experiment02a.{0}.{1}.xpck')

	# evaluate hierarchical model on training data
	loglik = mean(model.loglikelihood(data[:, :num_data])) + logjacobian
	print '{0} [bit/pixel]'.format(loglik / data.shape[0] / log(2.))

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
