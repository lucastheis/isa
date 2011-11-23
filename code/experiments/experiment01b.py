"""
Plot 
"""

import sys
import pdb

sys.path.append('./code')

from numpy import *
from numpy.random import *
from models import ISA, Distribution
from tools.mapp import mapp
from matplotlib.pyplot import *
from time import time

Distribution.VERBOSITY = 0

def main(argv):
	mapp.max_processes = 1

	m = ISA(81, 162)
#	m.initialize(method='gabor')
	m.initialize()

	Y = m.sample_prior(1000)
	X = dot(m.A, Y)

	start = time()

	target_energy = mean(m.energy(Y))

	for method, num_steps in [('gibbs', 20), ('hmc', 30)]:
		samples = [(start, m.sample_posterior(X, method=(method, 0)))]
		for t in range(num_steps):
			print t
			samples.append((
				time(), 
				m.sample_posterior(X, method=(method, 1), init=samples[-1][1])))

		energy = []
		for sample in samples:
			energy.append([sample[0] - start, mean(m.energy(sample[1]))])
		energy = vstack(energy).T

		plot(energy[0], energy[1])
		draw()

	plot([energy[0, 0], energy[0, -1]], [target_energy, target_energy], 'k--')
	legend(['Gibbs', 'HMC'])
	xlabel('seconds')
	ylabel('average energy')
	title('trace plot of random walk')

	pdb.set_trace()

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))

