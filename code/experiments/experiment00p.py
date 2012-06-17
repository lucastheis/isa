"""
AIS sanity check.
"""

import sys

sys.path.append('./code')

from models import ISA
from numpy import eye, sqrt, log, mean
from numpy.random import seed
from tools import logmeanexp

NUM_AIS_SAMPLES = 10
NUM_AIS_STEPS = 10

def main(argv):
	seed(42)

	isa1 = ISA(2)
	isa1.A = eye(2)

	for gsm in isa1.subspaces:
		gsm.scales[:] = 4.

	# equivalent overcomplete model
	isa2 = ISA(2, 4)
	isa2.A[:, :2] = isa1.A / sqrt(2.)
	isa2.A[:, 2:] = isa1.A / sqrt(2.)

	for gsm in isa2.subspaces:
		gsm.scales[:] = 4.

	data = isa1.sample(100)

	# the results should not depend on the parameters
	print -isa1.evaluate(data)
#	print -isa2.evaluate(data, num_samples=20, sampling_method=('ais', {'num_steps': 20}))

	ais_weights = isa2.loglikelihood(data,
		num_samples=NUM_AIS_SAMPLES, 
		sampling_method=('ais', {'num_steps': NUM_AIS_STEPS}), return_all=True)
	ais_weights = ais_weights# / log(2.) / data.shape[0]

	print mean(logmeanexp(ais_weights, 0)) / log(2.) / data.shape[0]

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
