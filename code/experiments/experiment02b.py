"""
Trains overcomplete ICA on a Gaussian scale mixture.
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

	# hyperparameters
	num_hiddens = int(argv[1]) if len(argv) > 1 else 2
	num_samples = int(argv[2]) if len(argv) > 2 else 400
	num_steps = int(argv[3]) if len(argv) > 3 else 20

	results = Experiment('results/experiment02a/experiment02a.xpck')

	# load Gaussian scale mixture
	gsm = results['gsm']

	# generate data
	data = results['data']

	# train overcomplete ICA model
	ica = ISA(gsm.dim, num_hiddens, 1)
	ica.train(data[:, :20000], max_iter=30, sampling_method=('Gibbs', {'num_steps': 20}))

	# evaluate model
	entropy = gsm.evaluate(data[:, -20000:])
	logloss = ica.evaluate(data[:, -20000:], num_samples=num_samples, num_steps=num_steps)

	# store results
	experiment['gsm'] = gsm
	experiment['ica'] = ica
	experiment['entropy'] = entropy
	experiment['logloss'] = logloss
	experiment.save('results/experiment02b/experiment02b.{{0}}.{{1}}.xpck'.format(num_hiddens))

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
