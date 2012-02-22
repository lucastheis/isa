"""
Train hierarchical ISA with manually picked subspace sizes in the first layers.
"""

import sys

sys.path.append('./code')

from numpy import *
from models import ConcatModel, StackedModel, Distribution, ISA, GSM
from transforms import LinearTransform, WhiteningTransform, RadialGaussianization
from transforms import SubspaceGaussianization, MarginalGaussianization
from tools import Experiment, preprocess, mapp

Distribution.VERBOSITY = 1

# number of processors used
mapp.max_processes = 8

def main(argv):
	seterr(invalid='raise', over='raise', divide='raise')

	# hyperparameters
	patch_size = '8x8'
	num_data = 100000
	max_iter = 40

	# start experiment
	experiment = Experiment()

	# load data, log-transform and center data
	data = load('data/vanhateren.{0}.preprocessed.npz'.format(patch_size))

	data_train = data['data_train']
	data_test = data['data_test']
	
	

	
	# train first layer
	gsm = GSM(data_train.shape[0])
	gsm.train(data_train, max_iter=100)

	# evaluate model
	print
	print '{0:.4f} [bit/pixel]'.format(
		mean(gsm.loglikelihood(data_train)) / data_train.shape[0] / log(2.))
	print '{0:.4f} [bit/pixel] (test)'.format(
		mean(gsm.loglikelihood(data_test)) / data_train.shape[0] / log(2.))
	print

	# transform data
	rg = RadialGaussianization(gsm)
	logjacobian = mean(rg.logjacobian(data_train))
	data_train = rg(data_train)



	# train second layer
	isa = ISA(data_train.shape[0], ssize=1)
	isa.initialize(method='laplace')
	isa.train(data_train[:, :20000], max_iter=20, method='sgd', train_prior=False)
	isa.train(data_train[:, :20000], max_iter=40, method='sgd', train_prior=True)
	isa.train(data_train, max_iter=10, method='lbfgs', train_prior=True)

	# evaluate model
	print
	print '{0:.4f} [bit/pixel]'.format(
		mean(isa.loglikelihood(data_train) + logjacobian) / data_train.shape[0] / log(2.))
	print

	# transform data
	sg = SubspaceGaussianization(isa)
	logjacobian += mean(sg.logjacobian(data_train))
	data_train = sg(data_train)




	# train third layer
	isa = ISA(data_train.shape[0], ssize=1)
	isa.initialize(method='laplace')
	isa.train(data_train[:, :20000], max_iter=20, method='sgd', train_prior=False)
	isa.train(data_train[:, :20000], max_iter=40, method='sgd', train_prior=True)
	isa.train(data_train, max_iter=10, method='lbfgs', train_prior=True)

	print
	print '{0:.4f} [bit/pixel]'.format(
		mean(isa.loglikelihood(data_train) + logjacobian) / data_train.shape[0] / log(2.))
	print



	# evaluate complete hierarchica model on test data
	model = StackedModel(rg, sg, isa)
	logloss = model.evaluate(data_test)

	print
	print '{0:.4f} [bit/pixel] (test)'.format(logloss)
	print



	# store results
	experiment['model'] = model
	experiment['logloss'] = logloss
	experiment.save('results/experiment02b/experiment02b.{0}.{1}.xpck')

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
