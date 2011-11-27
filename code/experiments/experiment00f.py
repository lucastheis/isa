import sys

sys.path.append('./code')

from models import ISA, GSM, Distribution
from tools import contours
from numpy.random import seed
from numpy import histogram
from matplotlib.pyplot import *

def main(argv):
	seed(3)

	# data dimensionality
	dim = 2

	# create Gaussian scale mixture
	gsm = GSM(dim, 20)
	gsm.initialize('student')

	# sample data
	data = gsm.sample(1000000)

	print gsm.evaluate(data[:, :10000])

	# train overcomplete ICA model
	ica = ISA(dim, argv[1] if len(argv) > 1 else 2, 1)
	ica.train(data[:, :10000], max_iter=20)

	print ica.evaluate(data[:, :10000])

	samples = ica.sample(1000000)

	# contour plots
	figure()
	contours(data[:2], 100)
	axis('equal')
	ax = axis()
	title('joint distribution (GSM)')

	figure()
	contours(samples[:2], 100)
	axis(ax)
	title('joint distribution (ICA)')

	# plot marginal distribution
	figure()
	h, x = histogram(data[1], 200, density=True)
	plot((x[1:] + x[:-1]) / 2., h)
	title('marginal distribution (GSM)')

	raw_input()

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
