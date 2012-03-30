#!/usr/bin/env python

"""
Use PCA filters and optimize marginals.
"""

import sys

sys.path.append('./code')

from numpy import *
from numpy.random import randn
from models import ISA, ICA, Distribution
from transforms import LinearTransform, WhiteningTransform
from tools import preprocess, Experiment, mapp

mapp.max_processes = 8
Distribution.VERBOSITY = 2

from numpy import round, sqrt, eye
from numpy.linalg import svd

patch_size = '16x16'
num_data = 100000
noise_level = 32

def main(argv):
	seterr(invalid='raise', over='raise', divide='raise')

	# start experiment
	experiment = Experiment(server='10.38.138.150')



	### DATA HANDLING

	# load data, log-transform and center data
	data = load('data/vanhateren.{0}.1.npz'.format(patch_size))['data']
	data = preprocess(data, noise_level=noise_level)
	
	# apply discrete cosine transform and remove DC component
	dct = LinearTransform(dim=int(sqrt(data.shape[0])), basis='DCT')
	data = dct(data)[1:]

	# PCA whitening
	wt = WhiteningTransform(data, symmetric=False)
	data = wt(data)



	### MODEL DEFINITION

	# create ICA model
	ica = ICA(data.shape[0])
	ica.initialize(method='laplace')
	ica.A = eye(data.shape[0])



	### TRAIN MODEL

	# train using L-BFGS
	ica.train(data[:, :num_data],
		max_iter=10,
		train_prior=True,
		train_basis=False)



	### EVALUATE MODEL

	data_test = load('data/vanhateren.{0}.0.npz'.format(patch_size))['data']
	data_test = preprocess(data_test, noise_level=noise_level)
	data_test = wt(dct(data_test)[1:])

	print 'training: {0:.4f} [bit/pixel]'.format(ica.evaluate(data))
	print 'test: {0:.4f} [bit/pixel]'.format(ica.evaluate(data_test))


#	# save results
#	experiment.save('results/experiment00o/experiment00o.{0}.{1}.xpck')

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
