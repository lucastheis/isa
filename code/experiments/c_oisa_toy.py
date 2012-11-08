#!/usr/bin/env python

"""
Test overcomplete ISA.
"""

import os
import sys

sys.path.append('./code')

from isa import ISA
from numpy import *
from numpy import round, max, min
from numpy.random import *
from tools import sqrtmi
from pprint import pprint

def main(argv):
	A = array([
		[1, 0, 0, 0, 0, 0, 1, 0],
		[0, 1, 0, 0, 1, 0, 0, 0],
		[0, 0, 1, 0, 0, 1, 0, 0],
		[0, 0, 0, 1, 0, 0, 0, 1]], dtype='float')

	isa1 = ISA(4, 8, ssize=2)
	isa1.initialize()
	isa1.A = A

	samples = isa1.sample(20000)

	def callback(i, isa):
		os.system('clear')
		print round(A, 2)
		print
		print round((abs(isa.A) > 0.5) * 1., 2)
		print
		print round(isa.A, 2)

	isa2 = ISA(4, 8, ssize=2)
	isa2.set_subspaces(isa1.subspaces())
	isa2.train(samples, parameters={
		'merge_subspaces': False,
		'train_prior': False,
		'training_method': 'lbfgs',
		'lbfgs': {'max_iter': 100, 'num_grad': 20},
		'gibbs': {'num_iter': 2},
		'callback': callback,
		'max_iter': 1000
	})

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
