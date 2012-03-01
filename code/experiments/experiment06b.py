"""
Learn filters using Olshausen & Field's sparse coding algorithm.
"""

import sys

sys.path.append('./code')

from models import ISA
from transforms import WhiteningTransform
from tools import preprocess, Experiment, mapp, patchutil
from numpy import load, std
from numpy.random import *

mapp.max_processes = 20

from matplotlib.pyplot import *
from numpy import cov

# PS, DS, OC
parameters = [
	# complete models
	['8x8',     'OF', 1],
	['8x8',   'VANH', 1],
	['16x16',   'OF', 1],
	['16x16', 'VANH', 1],

	# overcomplete models
	['8x8',     'OF', 2],
	['8x8',   'VANH', 2],
	['16x16',   'OF', 2],
	['16x16', 'VANH', 2],
]

def main(argv):
	if len(argv) < 2:
		print 'Usage:', argv[0], '<params_id>'
		return 0

	# parameters of the experiment
	patch_size, \
	dataset, \
	overcompleteness = parameters[int(argv[1])]

	experiment = Experiment()

	# load data, log-transform and center data
	if dataset == 'VANH':
		data = load('data/vanhateren.8x8.1.npz')['data']
		data = preprocess(data)

		# create whitening transform
		wt = WhiteningTransform(data, symmetric=True)
		data = wt(data)

		experiment['wt'] = wt

	elif dataset == 'OF':
		# load image patches used by Olshausen & Field
		data = load('data/of.8x8.npz')['data']
		data = data[:, permutation(data.shape[1])]

	# create model
	model = ISA(
		num_visibles=data.shape[0],
		num_hiddens=overcompleteness * data.shape[0],
		ssize=1)
	model.initialize(method='laplace')
	
	# initialize filters
	model.A = rand(*model.A.shape) - 0.5
	model.orthogonalize()

	if dataset == 'VANH':
		model.train_of(data,
			max_iter=100,
			noise_var=0.1,
			var_goal=1.,
			beta=10.,
			step_width=0.01,
			sigma=1.0)

	else:
		model.train_of(data,
			max_iter=100,
			noise_var=0.005,
			var_goal=0.1,
			beta=1.2,
			sigma=0.07)

	experiment['model'] = model
	experiment.save('results/experiment06b/experiment06b.{0}.{1}.xpck')

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
