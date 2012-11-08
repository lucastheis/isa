"""
Plot autocorrelation for toy model.
"""

import os
import sys

sys.path.append('./code')

from models import ISA, GSM, Distribution
from numpy import *
from numpy import max
from numpy.random import *
from numpy.linalg import pinv
from tools import contours, mapp
from pgf import *
from pdb import set_trace
from time import time
from tools import Experiment
from copy import deepcopy

NUM_SAMPLES = 5000 # used to estimate time transition operator takes
NUM_STEPS_MULTIPLIER = 5 # number of transition operator applications for estimating computation time
NUM_AUTOCORR = 5000 # number of posterior autocorrelation functions averaged
NUM_SECONDS_RUN = 10000 # length of Markov chain used to estimate autocorrelation
NUM_SECONDS_VIS = 15 # length of estimated autocorrelation function

# transition operator parameters
sampling_methods = [
	{
		'method': 'hmc', 
		'burn_in_steps': 20,
		'parameters': {
			'num_steps': 1,
			'lf_step_size': 1.25,
			'lf_num_steps': 5,
			'lf_randomness': 0.05,
		},
		'color': RGB(0.9, 0.0, 0.0),
	},
	{
		'method': 'mala', 
		'burn_in_steps': 20,
		'parameters': {
			'num_steps': 5,
			'step_width': 3.5,
		},
		'color': RGB(0.4, 0.2, 0.0),
	},
	{
		'method': 'gibbs',
		'burn_in_steps': 20,
		'parameters': {
			'num_steps': 1,
		},
		'color': RGB(0.1, 0.6, 1.),
	},
]



def autocorr(X, N, d=1):
	"""
	Estimates autocorrelation from a sample of a possibly multivariate
	stationary Markov chain.
	"""

	X = X - mean(X, 1).reshape(-1, 1)
	v = mean(sum(square(X), 0))

	# autocovariance
	A = [v]

	for t in range(1, N + 1, d):
		A.append(mean(sum(X[:, :-t] * X[:, t:], 0)))

	# normalize by variance
	return hstack(A) / v



def main(argv):
	seterr(over='raise', divide='raise', invalid='raise')

	try:
		if int(os.environ['OMP_NUM_THREADS']) > 1 or int(os.environ['MKL_NUM_THREADS']) > 1:
			print 'It seems that parallelization is turned on. This will skew the results. To turn it off:'
			print '\texport OMP_NUM_THREADS=1'
			print '\texport MKL_NUM_THREADS=1'
	except:
		print 'Parallelization of BLAS might be turned on. This could skew results.'

	experiment = Experiment(seed=42)

	if os.path.exists('results/toyexample/toyexample.xpck'):
		results = Experiment('results/toyexample/toyexample.xpck')
		ica = results['ica']
	else:
		# toy model
		ica = ISA(1, 3)
		ica.initialize(method='exponpow')
		ica.A = 1. + randn(1, 3) / 5.

		experiment['ica'] = ica
		experiment.save('results/toyexample/toyexample.xpck')

	Y_ = ica.sample_prior(NUM_AUTOCORR)
	X_ = dot(ica.A, Y_)

	for method in sampling_methods:
		# disable output and parallelization
		Distribution.VERBOSITY = 0
		mapp.max_processes = 1

		Y = ica.sample_prior(NUM_SAMPLES)
		X = dot(ica.A, Y)

		# measure time required by transition operator
		start = time()

		# increase number of steps to reduce overhead
		ica.sample_posterior(X, method=(method['method'], dict(method['parameters'],
			Y=Y, num_steps=method['parameters']['num_steps'] * NUM_STEPS_MULTIPLIER)))

		# time required per transition operator application
		duration = (time() - start) / NUM_STEPS_MULTIPLIER

		# number of mcmc steps to run for this method
		num_mcmc_steps = int(NUM_SECONDS_RUN / duration + 1.)
		num_autocorr_steps = int(NUM_SECONDS_VIS / duration + 1.)

		# enable output and parallelization
		Distribution.VERBOSITY = 2
		mapp.max_processes = 2

		# posterior samples
		Y = [Y_]

		# Markov chain
		for i in range(num_mcmc_steps):
			Y.append(ica.sample_posterior(X_, 
				method=(method['method'], dict(method['parameters'], Y=Y[-1]))))

		ac = []

		for j in range(NUM_AUTOCORR):
			# collect samples belonging to one posterior distribution
			S = hstack([Y[k][:, [j]] for k in range(num_mcmc_steps)])

			# compute autocorrelation for j-th posterior
			ac = [autocorr(S, num_autocorr_steps)]

		# average and plot autocorrelation functions
		plot(arange(num_autocorr_steps) * duration, mean(ac, 0), '-', 
			color=method['color'],
			line_width=1.2,
			comment=str(method['parameters']))

	xlabel('time in seconds')
	ylabel('autocorrelation')
	title('toy example')

	gca().width = 7
	gca().height = 7
	gca().xmin = -1
	gca().xmax = NUM_SECONDS_VIS

	savefig('results/toyexample/toyexample_autocorr2.tex')

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
