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

	num_hiddens = int(argv[1]) if len(argv) > 1 else 2
	num_samples = int(argv[2]) if len(argv) > 2 else 10
	num_steps = int(argv[3]) if len(argv) > 3 else 10

	seed(3)

	# data dimensionality
	dim = 2

	# create Gaussian scale mixture
	gsm = GSM(dim, 20)
	gsm.initialize('student')

	seed(int(time() * 1E4))

	# sample data
	data = gsm.sample(1000000)

	# train overcomplete ICA model
	ica = ISA(dim, num_hiddens, 1)
	ica.train(data[:, :10000], max_iter=30, sampling_method=('Gibbs', {'num_steps': 20}))

	entropy = gsm.evaluate(data[:, :20000])
	logloss = ica.evaluate(data[:, :20000], num_samples=num_samples, num_steps=num_steps)

	print entropy
	print logloss

	samples = ica.sample(1000000)

	try:
		# contour plots
		figure()
		contours(data[:2], 100)
		axis('equal')
		ax = axis()
		title('joint distribution (GSM)')
		savefig('results/experiment00f/j_gsm.png')

		figure()
		contours(samples[:2], 100)
		axis('equal')
		axis(ax)
		title('joint distribution (ICA)')
		savefig('results/experiment00f/j_ica.{0}.png'.format(num_hiddens))

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
		savefig('results/experiment00f/marginal.{0}.png'.format(num_hiddens))

	except:
		print 'Plotting failed.'

	experiment['gsm'] = gsm
	experiment['ica'] = ica
	experiment['entropy'] = entropy
	experiment['logloss'] = logloss
	experiment.save('results/experiment00f/experiment00f.{0}.{{0}}.{{1}}.xpck'.format(num_hiddens))

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
