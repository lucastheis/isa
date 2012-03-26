"""
Test calculation of the MAP.
"""

import sys

sys.path.append('./code')

from models import ISA
from pgf import *
from numpy import *

def main(argv):
	isa = ISA(1, 2)
	isa.A[:] = [0.8, 1.1]
	isa.initialize(method='student')

	X = zeros([1, 100]) + 9.# isa.sample(100)
	M = isa.compute_map(X)
	Y = isa.sample_prior(5000)

	plot(Y[0], Y[1], 'k.', opacity=0.1)
	plot(M[0], M[1], 'r.')
	axis([-20, 20, -20, 20])
	draw()

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
