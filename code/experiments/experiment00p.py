"""
AIS sanity check.
"""

import sys

sys.path.append('./code')

from models import ISA
from numpy import eye, sqrt

def main(argv):
	isa1 = ISA(2)
	isa1.A = eye(2)

	for gsm in isa1.subspaces:
		gsm.scales[:] = 1.

	# equivalent overcomplete model
	isa2 = ISA(2, 4)
	isa2.A[:, :2] = isa1.A / sqrt(2.)
	isa2.A[:, 2:] = isa1.A / sqrt(2.)

	for gsm in isa2.subspaces:
		gsm.scales[:] = 1.

	data = isa1.sample(100)

	# the results should not depend on the parameters
	print -isa1.evaluate(data)
	print -isa2.evaluate(data, num_samples=10, sampling_method=('ais', {'num_steps': 10}))

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
