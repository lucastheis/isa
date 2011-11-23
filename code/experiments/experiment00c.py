"""
Test training methods.
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

def main(argv):
	py_seed(1)
	np_seed(1)

	m = ISA(2, 4, 1)
	m.initialize()

	X = m.sample(1000)

	m = ISA(2, 4, 1)

	m.train(X)

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
