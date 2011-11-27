"""
Test sampling methods.
"""

import sys

sys.path.append('./code')

from numpy import *
from numpy import round, max, min
from numpy.random import *
from numpy.random import seed as np_seed
from numpy.linalg import *
from random import seed as py_seed
from models import ISA, GSM, Distribution
from tools.patchutil import show
from tools import contours
from tools.mapp import mapp
from pgf import *
from time import time

mapp.max_processes = 8

Distribution.VERBOSITY = 0

# sampling methods with parameters
methods = [
	('gibbs', {
		'num_steps': 20}),
	('hmc', {
		'num_steps': 100,
		'lf_step_size': 0.03,
		'lf_num_steps': 20}),
	('metropolis', {
		'num_steps': 5000,
		'standard_deviation': 0.03}),
	('ais', {
		'num_steps': 20}),
	]

def main(argv):
#	py_seed(1)
#	np_seed(1)

	m = ISA(4, 8, 2)

	Y = m.sample_prior(5000)

	# target energy
	print '\t{0:.2f}'.format(mean(m.prior_energy(Y)))
	print

	# sample posterior
	X = dot(m.A, Y)
	C = dot(m.A, m.A.T)
	logl = -0.5 * sum(multiply(X, dot(inv(C), X)), 0) \
		- 0.5 * slogdet(C)[1]  - m.num_visibles / 2. * log(2.  * pi)
	logl = logl.reshape(1, -1)

	Y, logw = m.sample_posterior_ais(X, num_steps=2)

	logl = m.loglikelihood(X, num_samples=40, num_steps=10)

	print
	print '\t{0:.2f}'.format(mean(exp(logw - logl) * m.prior_energy(Y)))

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
