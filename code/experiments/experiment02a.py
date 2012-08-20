"""
Train hierarchical ISA.
"""

import sys

sys.path.append('./code')

from numpy import *
from models import ConcatModel, StackedModel, Distribution, ISA, ICA
from transforms import LinearTransform, WhiteningTransform
from transforms import SubspaceGaussianization, MarginalGaussianization
from tools import Experiment, preprocess, mapp

Distribution.VERBOSITY = 1

# number of processors used
mapp.max_processes = 8

# PS, SS, ND, MI, NS, ML, LS, RG
parameters = [
	['8x8',   1, 100,  40,   32, 50, False, False],
	['16x16', 1, 100, 200,   32, 50, False, False],
	['8x8',   2, 100,  40,   32, 50, False, False],
	['16x16', 2, 100,  80,   32, 50, False, False],
	['8x8',   1, 100,  40,   32, 50, True,  False],
	['16x16', 1, 100,  40,   32, 50, True,  False],
	['8x8',   1, 100,  40,   32, 50, False, True],
	['8x8',   1, 100,  40, None, 50, False, False],
]

def main(argv):
	if len(argv) < 2:
		print 'Usage:', argv[0], '<params_id>'
		print
		print '  {0:>3} {1:>5} {2:>3} {3:>5} {4:>4} {5:>5} {6:>3} {7:>5} {8:>5}'.format(
			'ID', 'PS', 'SS', 'ND', 'MI', 'NS', 'ML', 'LS', 'RG')

		for id, params in enumerate(parameters):
			print '  {0:>3} {1:>5} {2:>3} {3:>4}k {4:>4} {5:>5} {6:>3} {7:>5} {8:>5}'.format(id, *params)

		print
		print '  ID = parameter set'
		print '  PS = patch size'
		print '  SS = subspace size'
		print '  ND = number of training points'
		print '  MI = maximum number of training epochs'
		print '  NS = inverse noise level'
		print '  ML = maximum number of layers'
		print '  LS = learn subspace sizes'
		print '  RG = radially Gaussianize first'

		return 0

	seterr(invalid='raise', over='raise', divide='raise')

	# hyperparameters
	patch_size, \
	ssize, \
	num_data, \
	max_iter, \
	noise_level, \
	max_layers, \
	train_subspaces, \
	radially_gaussianize = parameters[int(argv[1])]
	num_data = num_data * 1000

	# start experiment
	experiment = Experiment()

#	# load log-transformed and centered data
#	data = load('data/vanhateren.{0}.preprocessed.npz'.format(patch_size))
#
#	data_train = data['data_train'][:, :num_data]
#	data_test = data['data_test']

	# load data, log-transform and center data
	data_train = load('data/vanhateren.{0}.1.npz'.format(patch_size))['data']
	data_train = preprocess(data_train, noise_level=noise_level)

	data_test = load('data/vanhateren.{0}.0.npz'.format(patch_size))['data']
	data_test = preprocess(data_test, noise_level=noise_level, shuffle=False)[:, :10000]

	# container for hierarchical model
	model = []

	# average log-Jacobian determinant
	logjacobian = 0.

	experiment['parameters'] = parameters[int(argv[1])]
	experiment['model'] = model

	if radially_gaussianize:
		model.append(GSM(data_train.shape[0], 10))
		model[-1].train(data_train, max_iter=100)

		# evaluate GSM
		loglik_train = mean(model[-1].loglikelihood(data_train[:, :num_data]))
		loglik_test = mean(model[-1].loglikelihood(data_test[:, :num_data]))

		print
		print '{0:>2}, {1} [bit/pixel] (train)'.format(len(model), -loglik_train / data_train.shape[0] / log(2.))
		print '{0:>2}, {1} [bit/pixel] (test)'.format(len(model), -loglik_test / data_train.shape[0] / log(2.))
		print

		# perform radial Gaussianization
		rg = RadialGaussianization(model[-1])

		logjacobian += mean(rg.logjacobian(data_test[:, :num_data]))

		data_train = rg(data_train)
		data_test = rg(data_test)

	for _ in range(max_layers):
		model.append(ISA(data_train.shape[0], data_train.shape[0], ssize=ssize))
		
		# initialize, train and finetune model
		model[-1].initialize(method='laplace')
		model[-1].initialize(data_train)

		model[-1].train(data_train[:, :20000], max_iter=20, method=('sgd', {'max_iter': 1}),
				train_prior=False, train_subspaces=False)

		model[-1].train(data_train[:, :20000], max_iter=max_iter, method=('sgd', {'max_iter': 1}),
				train_prior=True, train_subspaces=train_subspaces)

		model[-1].train(data_train[:, :num_data], max_iter=25, method='lbfgs',
				train_prior=True, train_subspaces=train_subspaces)

		# evaluate hierarchical model on training and test data
		loglik_train = model[-1].loglikelihood(data_train[:, :num_data]) + logjacobian
		loglik_test = model[-1].loglikelihood(data_test[:, :num_data]) + logjacobian

		print
		print '{0:>2}, {1} [bit/pixel] (train)'.format(len(model), -mean(loglik_train) / data_train.shape[0] / log(2.))
		print '{0:>2}, {1} [bit/pixel] (test)'.format(len(model), -mean(loglik_test) / data_train.shape[0] / log(2.))
		print

		# save model
		experiment['num_layers'] = len(model)
		experiment['loglik_train'] = loglik_train
		experiment['loglik_test'] = loglik_test
		experiment.save('results/experiment02a/experiment02a.{0}.{1}.xpck')

		# perform subspace Gaussianization on data and update Jacobian
		sg = SubspaceGaussianization(model[-1])

		logjacobian += mean(sg.logjacobian(data_test[:, :num_data]))

		data_train = sg(data_train[:, :num_data])
		data_test = sg(data_test[:, :num_data])

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
