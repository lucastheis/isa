"""
Visualize random walk in null space.
"""

import sys

sys.path.append('./code')

from models import ISA, GSM, Distribution
from numpy import *
from numpy.random import *
from tools import contours, mapp
from matplotlib.pyplot import *
from pdb import set_trace
from time import time

Distribution.VERBOSITY = 2
mapp.max_processes = 1

parameters = {
	'gibbs': {
		'num_steps': 1,
	},
	'tempered': {
		'num_steps': 1,
		'annealing_weights': arange(0.8, 1., 0.05)
	},
	'hmc': {
		'num_steps': 10,
		'lf_step_size': 1.,
		'lf_num_steps': 20
	},
}

def main(argv):
	seed(4)

	## SAMPLING METHOD

	if len(argv) > 1:
		if argv[1] not in parameters:
			raise ValueError('Unknown sampling method ''{0}'''.format(argv[1]))
		sampling_method = argv[1]
	else:
		sampling_method = 'gibbs'


	## MODEL

	isa = ISA(1, 3)
	isa.initialize(method='student')

	isa.A[:] = 1.
	isa.A += randn(isa.num_visibles, isa.num_hiddens) / 3.



	## POSTERIOR SAMPLES

	# visible state
	X = zeros([isa.num_visibles, 1]) + 18.

	samples = isa.sample_nullspace(repeat(X, 20000, 1), method=('gibbs', {'num_steps': 20}))



	## MCMC TRACE

	random_walk = [isa.sample_posterior(X, method=(sampling_method, {'num_steps': 0}))]

	for _ in range(20):
		random_walk.append(isa.sample_posterior(X,
			method=(sampling_method, dict(parameters[sampling_method], Y=random_walk[-1]))))
	random_walk = dot(isa.nullspace_basis(), hstack(random_walk))



	## VISUALIZE

	clf()
	plot(random_walk[0], random_walk[1], 'rs--', markeredgewidth=0)
	contours(samples, 60, 0.04 * 0.6**arange(7), colors='k')
	axis('equal')
	axis('off')
	axis([-20, 20, -20, 20])
	draw()
	savefig('/Users/lucas/Desktop/contours.png')

	raw_input()

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
