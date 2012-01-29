"""
Visualize random walk in null space.
"""

import sys

sys.path.append('./code')

from models import ISA, GSM, Distribution
from numpy import *
from numpy.random import *
from numpy.linalg import pinv
from tools import contours, mapp
from pgf import *
from pdb import set_trace
from time import time

Distribution.VERBOSITY = 2
mapp.max_processes = 1

parameters = {
	'gibbs': {
		'num_steps': 4,
	},
	'tempered': {
		'num_steps': 1,
		'annealing_weights': arange(0.8, 1., 0.05)
	},
	'hmc': {
		'num_steps': 4,
		'lf_step_size': 1.,
		'lf_num_steps': 10
	},
}

titles = {
	'gibbs': 'Gibbs sampling',
	'hmc': 'HMC sampling',
	'tempered': 'Gibbs sampling with tempered transitions'
}

def main(argv):
	seterr(over='raise', divide='raise', invalid='raise')
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
	isa.A += randn(3) / 5.



	## MCMC TRACE

	# visible state
	X = zeros([isa.num_visibles, 1]) + 12.

	random_walk = [isa.sample_posterior(X, method=('gibbs', {'num_steps': 20}))]
	random_walk = [dot(pinv(isa.A), X)]

	for _ in range(40):
		random_walk.append(isa.sample_posterior(X,
			method=(sampling_method, dict(parameters[sampling_method], Y=random_walk[-1]))))
	random_walk = dot(isa.nullspace_basis(), hstack(random_walk))



	## VISUALIZE

	Z1, Z2 = meshgrid(linspace(-20, 20, 128), linspace(-20, 20, 128))
	Z = vstack([Z1.flatten(), Z2.flatten()])
	Y = dot(pinv(isa.A), X) + dot(pinv(isa.nullspace_basis()), Z)
	E = isa.prior_energy(Y).reshape(*Z1.shape)[::-1]

	imshow(exp(-E), limits=[-20, 20, -20, 20])
	plot(random_walk[0], random_walk[1], 'w.--', markeredgewidth=0)
	xlabel('$z_1$')
	ylabel('$z_2$')
	title(titles[sampling_method])
	gca().width = 10
	gca().height = 10
	draw()

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
