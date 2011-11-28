"""
Creates a random Gaussian scale mixture.
"""

import sys

sys.path.append('./code')

from models import ISA, GSM, Distribution
from tools import contours, Experiment
from numpy.random import seed
from numpy import histogram
from matplotlib.pyplot import *
from time import time

def main(argv):
	experiment = Experiment()

	seed(3)

	# create Gaussian scale mixture
	gsm = GSM(2, 20)
	gsm.initialize('student')

	# store results
	experiment['gsm'] = gsm
	experiment.save('results/experiment02a/experiment02a.xpck')

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
