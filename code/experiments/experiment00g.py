"""
Test recovery of subspaces in overcomplete ISA.
"""

import sys

sys.path.append('./code')

from models import ISA, Distribution
from numpy import *
from pgf import *
from tools import contours, sqrtmi
from copy import deepcopy

Distribution.VERBOSITY = 2

def main(argv):
	isa1 = ISA(2, 3, ssize=3)
	isa1.A = dot(sqrtmi(dot(isa1.A, isa1.A.T)), isa1.A)
	isa1.initialize(method='laplace')

	samples = isa1.sample(10000)

	subplot(0, 0)
	plot(samples[0], samples[1], 'b.', opacity=0., marker_opacity=0.1)
	axis('equal')
	axis([-8, 8, -8, 8])
	title('true model')

	isa2 = ISA(2, 3, ssize=1)
	isa2.train(samples, max_iter=10,
		train_prior=False,
		train_subspaces=False,
		persistent=True,
		method=('sgd', {'max_iter': 1}),
		sampling_method=('gibbs', {'num_steps': 5}))
	isa2.train(samples, max_iter=40,
		train_prior=True,
		train_subspaces=True,
		persistent=True,
		method=('sgd', {'max_iter': 1}),
		sampling_method=('gibbs', {'num_steps': 5}))

	samples = isa2.sample(10000)

	print [s.dim for s in isa1.subspaces]
	print [s.dim for s in isa2.subspaces]

	subplot(0, 1)
	plot(samples[0], samples[1], 'r.', opacity=0., marker_opacity=0.1)
	axis('equal')
	axis([-8, 8, -8, 8])
	title('recovered model')
	draw()

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
