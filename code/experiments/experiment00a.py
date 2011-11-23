"""
Test sampling methods.
"""

import sys

sys.path.append('./code')

from numpy import *
from numpy.random import *
from numpy.random import seed as np_seed
from random import seed as py_seed
from models import ISA, GSM, Distribution
from tools.patchutil import show
from tools import contours
from tools.mapp import mapp
from matplotlib.pyplot import *
from time import time

Distribution.VERBOSITY = 0

# sampling methods with parameters
methods = [
	('gibbs', {
		'num_steps': 10}),
	('hmc', {
		'num_steps': 100,
		'lf_step_size': 0.03,
		'lf_num_steps': 20}),
	('metropolis', {
		'num_steps': 10000,
		'standard_deviation': 0.03}),
	]

def main(argv):
	py_seed(1)
	np_seed(1)

	m = ISA(2, 4, 1)
	m.initialize()

	Y = m.sample_prior(5000)

	# target energy
	print '\t{0:.2f}'.format(mean(m.energy(Y)))
	print

	# plot prior
	plot(Y[0], Y[1], '.')

	# sample posterior
	X = dot(m.A, Y)
	t = time()
	Y = m.sample_posterior(X, method=methods[1])

	print
	print '{0:.2f} seconds'.format(time() - t)

	# plot posterior
	plot(Y[0], Y[1], 'r.')
	axis('equal')
	axis([-5, 5, -5, 5])

	raw_input()

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
