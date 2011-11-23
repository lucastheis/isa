"""
Test GSM training.
"""

import sys

sys.path.append('./code')

from models import GSM
from scipy.stats import t
from pgf import *
from numpy import *

def main(argv):
	gsm = GSM(1, 4)

	v = 8.

	data = t.rvs(v, size=[1, 1000])

#	gsm.initialize('cauchy')
	gsm.train(data)

	x = linspace(-20, 20, 400)
	plot(x, t.pdf(x, v), 'b')
	plot(x, exp(gsm.loglikelihood(x.reshape(1, -1))), 'r')
	legend('Student-t', 'GSM')
	draw()


	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
