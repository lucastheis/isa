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
NUM_STEPS_MULTIPLIER = 5 # number of transition operator applications for estimating times
NUM_AUTOCORR = 500 # number of posterior autocorrelation functions averaged
NUM_CHAINS = 500 # number of chains used to estimate each autocorrelation function
NUM_SECONDS = 15

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

def main(argv):
	seterr(over='raise', divide='raise', invalid='raise')

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
			Y = repeat(ica.sample_prior(), NUM_CHAINS, 1)
			X = dot(ica.A, Y)

			# burn-in phase
			Y = ica.sample_posterior(X, method=(method['method'], dict(method['parameters'],
				Y=Y, num_steps=method['parameters']['num_steps'] * method['burn_in_steps'])))

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
	title('toy example')

	gca().width = 7
	gca().height = 7
	gca().xmin = -1
	gca().xmax = NUM_SECONDS

	savefig('results/toyexample/toyexample_autocorr.tex')

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
