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
from tools import contours, mapp, preprocess
from pgf import *
from pdb import set_trace
from time import time
from tools import Experiment
from copy import deepcopy

NUM_SAMPLES = 200 # used to estimate time transition operator takes
NUM_STEPS_MULTIPLIER = 5 # number of transition operator applications for estimating times
NUM_AUTOCORR = 200 # number of posterior autocorrelation functions averaged
NUM_CHAINS = 500 # number of chains used to estimate each autocorrelation function
NUM_SECONDS = 60
NUM_BURN_IN_STEPS = 100

EXPERIMENT_PATH = 'results/vanhateren/vanhateren.7.08042012.150147.xpck'

# transition operator parameters
sampling_methods = [
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 1,
			'lf_step_size': 0.075,
			'lf_num_steps': 10,
			'lf_randomness': 0.1,
		},
		'color': RGB(0.9, 0.0, 0.0),
	},
	{
		'method': 'mala', 
		'parameters': {
			'num_steps': 2,
			'step_width': 0.1,
		},
		'color': RGB(0.4, 0.2, 0.0),
	},
	{
		'method': 'gibbs',
		'parameters': {
			'num_steps': 2,
		},
		'color': RGB(0.1, 0.6, 1.),
	},
]

def main(argv):
	seterr(over='raise', divide='raise', invalid='raise')

	experiment = Experiment(seed=42)

	if not os.path.exists(EXPERIMENT_PATH):
		print 'Could not find file \'{0}\'.'.format(EXPERIMENT_PATH)
		return 0

	results = Experiment(EXPERIMENT_PATH)
	ica = results['model'].model[1].model

	# load test data
	data = load('data/vanhateren.{0}.0.npz'.format(results['parameters'][0]))['data']
	data = data[:, :100000]
	data = preprocess(data)
	data = data[:, permutation(data.shape[1] / 2)[:NUM_SAMPLES]]

	# transform data
	dct = results['model'].transforms[0]
	wt = results['model'].model[1].transforms[0]
	data = wt(dct(data)[1:])

	for method in sampling_methods:
		# disable output and parallelization
		Distribution.VERBOSITY = 0
		mapp.max_processes = 1

		# measure time required by transition operator
		start = time()

		Y = ica.sample_prior(NUM_SAMPLES)
		X = dot(ica.A, Y)

		# increase number of steps to reduce overhead
		ica.sample_posterior(X, method=(method['method'], dict(method['parameters'],
			Y=Y, num_steps=method['parameters']['num_steps'] * NUM_STEPS_MULTIPLIER)))

		# time required per transition operator application
		duration = (time() - start) / NUM_STEPS_MULTIPLIER

		num_mcmc_steps = int(NUM_SECONDS / duration + 1.)

		# enable output and parallelization
		Distribution.VERBOSITY = 2
		mapp.max_processes = 2

		autocorr = []

		for n in range(NUM_AUTOCORR):
			# burn-in phase
			X = repeat(data[:, [n]], NUM_CHAINS, 1)
			Y = ica.sample_posterior(X, method=('gibbs', {'num_steps': NUM_BURN_IN_STEPS}))

			energies = [ica.prior_energy(Y)]

			# Markov chain
			for i in range(num_mcmc_steps):
				Y = ica.sample_posterior(X, method=(method['method'], 
					dict(method['parameters'], Y=Y)))
				energies.append(ica.prior_energy(Y))

			energies = vstack(energies)

			energy_mean = mean(energies)
			energy_var = mean(square(energies[[0]] - energy_mean))
			energy_cov = mean((energies - energy_mean) * (energies[[0]]  - energy_mean), 1)
			
			autocorr.append(energy_cov / energy_var)

		plot(arange(num_mcmc_steps) * duration, mean(autocorr, 0), '-', color=method['color'],
			line_width=1.2, comment=str(method['parameters']))

	xlabel('time in seconds')
	ylabel('autocorrelation')
	title('van Hateren')

	gca().width = 7
	gca().height = 7
	gca().xmin = -1
	gca().xmax = NUM_SECONDS

	savefig('results/vanhateren/vanhateren_autocorr.tex')

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
