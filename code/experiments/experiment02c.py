"""
First train ICA, then radially Gaussianize.
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
	isa = ISA(data_train.shape[0], ssize=1)
	isa.initialize(method='laplace')
	isa.train(data_train[:, :20000], max_iter=10, method='sgd', train_prior=False)
	isa.train(data_train[:, :20000], max_iter=40, method='sgd', train_prior=True)
	isa.train(data_train, max_iter=10, method='lbfgs', train_prior=True)

	# evaluate model
	print
	print '{0:.4f} [bit/pixel] (test)'.format(isa.evaluate(data_test))
	print

	# subspace Gausianization transform
	sg = SubspaceGaussianization(isa)



	# train second layer
	gsm = GSM(data_train.shape[0])
	gsm.train(sg(data_train), max_iter=100)



	# evaluate model
	model = StackedModel(sg, gsm)

	print
	print '{0:.4f} [bit/pixel] (test)'.format(model.evaluate(data_test))
	print



	# store results
	experiment['model'] = model
	experiment.save('results/experiment02c/experiment02c.{0}.{1}.xpck')

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
