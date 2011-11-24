"""
Test GSM training.
"""

import sys

sys.path.append('./code')

from models import GSM
from scipy.stats import t
from pgf import *
from numpy import *
from scipy import integrate

def main(argv):
	# Student-t parameter
	v = 8.

	# sample from Student-t
#	data = t.rvs(v, size=[1, 2000])

	# train GSM
	gsm1 = GSM(1, 2)
#	gsm1.initialize('cauchy')
#	gsm1.train(data, max_iter=100)
	gsm1.scales = asarray([0.5, 4.])

	data = gsm1.sample(20000)

	gsm2 = GSM(1, 2)
	gsm2.train(data, max_iter=100)

	print gsm2.scales

	x = linspace(-20, 20, 400)
#	plot(x, t.pdf(x, v), 'k')
	plot(x, exp(gsm1.loglikelihood(x.reshape(1, -1))), 'r')
	plot(x, exp(gsm2.loglikelihood(x.reshape(1, -1))), 'b')
	legend('GSM1', 'GSM2', 'histogram')
	draw()


	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
