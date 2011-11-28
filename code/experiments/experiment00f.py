import sys

sys.path.append('./code')

from models import ISA, GSM, Distribution
from tools import contours, Experiment
from numpy.random import seed
from numpy import histogram
from matplotlib.pyplot import *

def main(argv):
	experiment = Experiment(seed=3)

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
	ica = ISA(dim, int(argv[1]) if len(argv) > 1 else 2, 1)
	ica.train(data[:, :10000], max_iter=20, sampling_method=('Gibbs', {'num_steps': 10}))

	print ica.evaluate(data[:, :10000])

	samples = ica.sample(1000000)

	# contour plots
	figure()
	contours(data[:2], 100)
	axis('equal')
	ax = axis()
	title('joint distribution (GSM)')
	savefig('j_gsm.png')

	figure()
	contours(samples[:2], 100)
	axis('equal')
	axis(ax)
	title('joint distribution (ICA)')
	savefig('j_ica.png')

	figure()
	contours(samples[:2], 100)

	# plot marginal distribution
	figure()
	h, x = histogram(data[1], 200, normed=True)
	plot((x[1:] + x[:-1]) / 2., h, 'b')
	h, x = histogram(samples[1], 200, normed=True)
	plot((x[1:] + x[:-1]) / 2., h, 'r')
	title('marginal distribution')
	legend(['GSM', 'ISA'])
	savefig('marginal.png')

	experiment['gsm'] = gsm
	experiment['ica'] = ica
#	experiment.save('results/experiment00f/experiment00f.xpck')

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
