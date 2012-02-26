"""
Test OF algorithm.
"""

import sys

sys.path.append('./code')

from models import ISA
from transforms import WhiteningTransform
from tools import preprocess, Experiment, mapp, patchutil
from numpy import load, std
from numpy.random import *

mapp.max_processes = 20

def main(argv):
	experiment = Experiment()

	# load data, log-transform and center data
#	data = load('data/vanhateren.8x8.1.npz')['data']
#	data = preprocess(data)
	data = load('data/of.8x8.npz')['data']
#	data = data / std(data.flatten())

	# create whitening transform
#	wt = WhiteningTransform(data, symmetric=True)
#	data = wt(data)

	isa = ISA(
		num_visibles=data.shape[0],
		num_hiddens=1. * data.shape[0],
		ssize=1)
	isa.A = rand(*isa.A.shape) - 0.5
	isa.orthogonalize()
	isa.train_of(data, max_iter=100)

	experiment['wt'] = wt
	experiment['isa'] = isa
	experiment.save('results/experiment01j/experiment01j.{0}.{1}.xpck')

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
