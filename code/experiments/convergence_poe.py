"""
Check for convergence of AIS estimate of PoE partition function.
"""

import sys

sys.path.append('./code')

from numpy import *
from numpy import min
from numpy.random import permutation
from scipy.io.matlab import loadmat
from pgf import *
from tools import Experiment, logmeanexp
from glob import glob

def main(argv):
	dim = 64
	imidx = 7

	# load unnormalized log-likelihood
	results = loadmat('results/vanhateren/poe/AIS_GibbsTrain_white_studentt_L=064_M=256_B=0100000_learner=PMPFdH1_20120523T112539.mat')
	loglik = -mean(results['E'][:, :10000]) - results['logZ']

	# load importance weights for partition function
	ais_weights = loadmat('results/vanhateren/poe/matlab_up=022150_T=10000000_ais.mat')['logweights']
	ais_weights.shape

	# number of samples to probe
	num_samples = 2**arange(0, ceil(log2(ais_weights.shape[0])) + 1, dtype='int32')
	num_samples[-1] = max([num_samples[-1], ais_weights.shape[0]])
	num_repetitions = ceil(2.**16 / num_samples)
	estimates = []

	print loadmat('results/vanhateren/poe/matlab_up=022150_T=10000000_ais.mat')['t_range'][:, imidx], 'intermediate distributions'

	logZ = logmeanexp(ais_weights[:, -1])

	for k in arange(len(num_samples)):
		estimates_ = []

		for _ in arange(num_repetitions[k]):
			# pick samples at random
			idx = permutation(ais_weights.shape[0])[:num_samples[k]]

			# estimate log-partf. using num_samples[k] samples
			loglik_ = loglik + (logZ - logmeanexp(ais_weights[idx, imidx]))

			# store estimate of log-likelihood 
			estimates_.append(loglik_)

		estimates.append(mean(estimates_))

	gca().width = 5
	gca().height = 5
#	gca().ymin = 0.85
#	gca().ymax = 1.55
#	ytick([0.9, 1.1, 1.3, 1.5])
	semilogx(num_samples, estimates / log(2.) / dim, '.-')
	xlabel('number of AIS samples')
	ylabel('estimated log-likelihood')
	savefig('results/vanhateren/convergence_poe.tex')
	draw()

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
