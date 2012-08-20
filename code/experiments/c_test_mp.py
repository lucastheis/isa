#!/usr/bin/env python

"""
Train overcomplete ICA/ISA on van Hateren image patches.
"""

import sys

sys.path.append('./code')

from models import MoGaussian, StackedModel, ConcatModel, Distribution
from isa import ISA, GSM
from tools import preprocess, Experiment, mapp, imsave, imformat, stitch
from transforms import LinearTransform, WhiteningTransform, RadialGaussianization
from numpy import seterr, sqrt, dot, load, hstack, eye, any, isnan
from numpy.random import rand
from numpy.linalg import inv

# controls parallelization
mapp.max_processes = 1

# controls how much information is printed during training
Distribution.VERBOSITY = 2

def main(argv):
	seterr(invalid='raise', over='raise', divide='raise')

	# start experiment
	experiment = Experiment()

	# hyperparameters
	patch_size = '16x16'
	overcompleteness = 5
	ssize = 2
	max_iter = 100
	num_coeff = 10
	num_visibles = 100



	### DATA PREPROCESSING

	# load data, log-transform and center data
	data = load('data/vanhateren.{0}.1.npz'.format(patch_size))['data']
	data = data[:, :100000]
	data = preprocess(data, shuffle=False)

	# discrete cosine transform and whitening transform
	dct = LinearTransform(dim=int(sqrt(data.shape[0])), basis='DCT')
	wt = WhiteningTransform(dct(data)[1:], symmetric=False)



	### MODEL DEFINITION

	isa = ISA(
		num_visibles=num_visibles,
		num_hiddens=num_visibles * overcompleteness, 
		ssize=ssize, 
		num_scales=20)

	# model DC component with a mixture of Gaussians
	model = StackedModel(dct,
		ConcatModel(
			MoGaussian(20), 
			StackedModel(wt, 
				ConcatModel(
					isa, 
					GSM(data.shape[0] - 1 - isa.dim)))))


	wt_sym = WhiteningTransform(data, symmetric=True)

	### MODEL TRAINING

	rec = dot(dct.A[1:].T, inv(wt.A)[:, :isa.dim])

	def callback(iteration, isa):
		"""
		Saves intermediate results every few iterations.
		"""

		if any(isnan(isa.subspaces()[0].scales)):
			print 'Scales are NaN.'
			return False

		if not iteration % 1:
			# whitened filters
			A = wt_sym(dot(rec, isa.A))

			patch_size = int(sqrt(model.dim) + 0.5)

			try:
				# visualize basis
				imsave('results/c_test_mp/basis.{0:0>3}.png'.format(iteration),
					stitch(imformat(A.T.reshape(-1, patch_size, patch_size), perc=99)))
			except:
				print 'Could not save intermediate results.'

	# initialize filters and marginals
	model.initialize(data, [1, 0])

	# initialize with matching pursuit
	model.train(data, [1, 0], parameters={
		'training_method': 'mp',
		'mp': {
			'max_iter': max_iter,
			'step_width': 0.01,
			'momentum': 0.8,
			'batch_size': 200,
			'num_coeff': num_coeff},
		'callback': callback})

	experiment.save('results/c_test_mp/c_test_mp.{0}.{1}.xpck')

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
