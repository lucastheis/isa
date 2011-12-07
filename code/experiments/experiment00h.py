import sys

sys.path.append('./code')

from models import ISA
from numpy import *

from tools import contours
from matplotlib.pyplot import *
from numpy.random import *


def main(argv):
	seterr(divide='raise', over='raise', invalid='raise')

	seed(2)

	isa = ISA(16, 32)

	data = isa.sample(1000)

	print isa.evaluate(data, num_samples=10, num_steps=10, method='biased')
	print isa.evaluate(data, num_samples=20, num_steps=10, method='biased')
	print

	return 0

	l = []

	for i in range(1000):
		l.append(isa.evaluate(data, num_steps=20, method='unbiased'))
		print '{0:8.4f} +- {1:.4f}, {2:6.4f}'.format(mean(l), sem(l), median(l))

	return 0



def sem(values):
	return std(values, ddof=1) / sqrt(len(values))



if __name__ == '__main__':
	sys.exit(main(sys.argv))
