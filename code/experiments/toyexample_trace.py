"""
Plot energy trace for toy model.
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

NUM_SAMPLES = 5000
NUM_SECONDS = 15
NUM_STEPS_MULTIPLIER = 5

# transition operator parameters
sampling_methods = [
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 2,
			'lf_step_size': 3.,
			'lf_num_steps': 2,
			'lf_randomness': 0.1,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 2,
			'lf_step_size': 2.5,
			'lf_num_steps': 2,
			'lf_randomness': 0.1,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 2,
			'lf_step_size': 2.,
			'lf_num_steps': 2,
			'lf_randomness': 0.1,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 2,
			'lf_step_size': 1.5,
			'lf_num_steps': 2,
			'lf_randomness': 0.1,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 2,
			'lf_step_size': 1.0,
			'lf_num_steps': 2,
			'lf_randomness': 0.1,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 2,
			'lf_step_size': 0.5,
			'lf_num_steps': 2,
			'lf_randomness': 0.1,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 1,
			'lf_step_size': 2.5,
			'lf_num_steps': 5,
			'lf_randomness': 0.1,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 1,
			'lf_step_size': 2.,
			'lf_num_steps': 5,
			'lf_randomness': 0.1,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 1,
			'lf_step_size': 1.5,
			'lf_num_steps': 5,
			'lf_randomness': 0.1,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 1,
			'lf_step_size': 1.25,
			'lf_num_steps': 5,
			'lf_randomness': 0.1,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 1,
			'lf_step_size': 1.0,
			'lf_num_steps': 5,
			'lf_randomness': 0.1,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 1,
			'lf_step_size': 0.75,
			'lf_num_steps': 5,
			'lf_randomness': 0.1,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 1,
			'lf_step_size': 1.25,
			'lf_num_steps': 5,
			'lf_randomness': 0.05,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 1,
			'lf_step_size': 1.0,
			'lf_num_steps': 5,
			'lf_randomness': 0.05,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 1,
			'lf_step_size': 0.75,
			'lf_num_steps': 5,
			'lf_randomness': 0.05,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 1,
			'lf_step_size': 1.25,
			'lf_num_steps': 5,
			'lf_randomness': 0.01,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 1,
			'lf_step_size': 1.0,
			'lf_num_steps': 5,
			'lf_randomness': 0.01,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 1,
			'lf_step_size': 0.75,
			'lf_num_steps': 5,
			'lf_randomness': 0.01,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 1,
			'lf_step_size': 0.5,
			'lf_num_steps': 5,
			'lf_randomness': 0.1,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 1,
			'lf_step_size': 0.25,
			'lf_num_steps': 5,
			'lf_randomness': 0.1,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 1,
			'lf_step_size': 1.5,
			'lf_num_steps': 10,
			'lf_randomness': 0.1,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 1,
			'lf_step_size': 1.0,
			'lf_num_steps': 10,
			'lf_randomness': 0.1,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 1,
			'lf_step_size': 0.75,
			'lf_num_steps': 10,
			'lf_randomness': 0.1,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 1,
			'lf_step_size': 0.5,
			'lf_num_steps': 10,
			'lf_randomness': 0.1,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 1,
			'lf_step_size': 0.25,
			'lf_num_steps': 10,
			'lf_randomness': 0.1,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 1,
			'lf_step_size': 1.0,
			'lf_num_steps': 20,
			'lf_randomness': 0.1,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 1,
			'lf_step_size': 0.75,
			'lf_num_steps': 20,
			'lf_randomness': 0.1,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 1,
			'lf_step_size': 0.5,
			'lf_num_steps': 20,
			'lf_randomness': 0.1,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 1,
			'lf_step_size': 0.25,
			'lf_num_steps': 20,
			'lf_randomness': 0.1,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'mala', 
		'parameters': {
			'num_steps': 5,
			'step_width': 4.,
		},
		'color': RGB(0.8, 0.8, 0.8),
	},
	{
		'method': 'mala', 
		'parameters': {
			'num_steps': 5,
			'step_width': 3.75,
		},
		'color': RGB(0.8, 0.8, 0.8),
	},
	{
		'method': 'mala', 
		'parameters': {
			'num_steps': 5,
			'step_width': 3.5,
		},
		'color': RGB(0.8, 0.8, 0.8),
	},
	{
		'method': 'mala', 
		'parameters': {
			'num_steps': 5,
			'step_width': 3.25,
		},
		'color': RGB(0.8, 0.8, 0.8),
	},
	{
		'method': 'mala', 
		'parameters': {
			'num_steps': 5,
			'step_width': 3.,
		},
		'color': RGB(0.8, 0.8, 0.8),
	},
	{
		'method': 'mala', 
		'parameters': {
			'num_steps': 5,
			'step_width': 2.75,
		},
		'color': RGB(0.8, 0.8, 0.8),
	},
	{
		'method': 'mala', 
		'parameters': {
			'num_steps': 5,
			'step_width': 2.5,
		},
		'color': RGB(0.8, 0.8, 0.8),
	},
	{
		'method': 'mala', 
		'parameters': {
			'num_steps': 5,
			'step_width': 2.,
		},
		'color': RGB(0.8, 0.8, 0.8),
	},
	{
		'method': 'mala', 
		'parameters': {
			'num_steps': 5,
			'step_width': 1.5,
		},
		'color': RGB(0.8, 0.8, 0.8),
	},
	{
		'method': 'mala', 
		'parameters': {
			'num_steps': 5,
			'step_width': 1.,
		},
		'color': RGB(0.8, 0.8, 0.8),
	},
	{
		'method': 'mala', 
		'parameters': {
			'num_steps': 5,
			'step_width': 0.5,
		},
		'color': RGB(0.8, 0.8, 0.8),
	},
	{
		'method': 'gibbs',
		'parameters': {
			'num_steps': 1,
		},
		'color': RGB(0.1, 0.6, 1.),
	},
]

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

	# generate visible and corresponding hidden states
	Y = ica.sample_prior(NUM_SAMPLES)
	X = dot(ica.A, Y)

	# energy of posterior samples should be around this value
	energy = mean(ica.prior_energy(Y))

	for method in sampling_methods:
		# disable output and parallelization
		Distribution.VERBOSITY = 0
		mapp.max_processes = 1

		# measure time required by transition operator
		start = time()

		# initial hidden states
		Y = dot(pinv(ica.A), X)

		# increase number of steps to reduce overhead
		ica.sample_posterior(X, method=(method['method'], 
			dict(method['parameters'], Y=Y, 
				num_steps=method['parameters']['num_steps'] * NUM_STEPS_MULTIPLIER)))

		# time required per transition operator application
		duration = (time() - start) / NUM_STEPS_MULTIPLIER

		# enable output and parallelization
		Distribution.VERBOSITY = 2
		mapp.max_processes = 2

		energies = [mean(ica.prior_energy(Y))]

		# Markov chain
		for i in range(int(NUM_SECONDS / duration + 1.)):
			Y = ica.sample_posterior(X,
				method=(method['method'], dict(method['parameters'], Y=Y)))
			energies.append(mean(ica.prior_energy(Y)))

		plot(arange(len(energies)) * duration, energies, '-', color=method['color'],
			line_width=1.2, pgf_options=['forget plot'], comment=str(method['parameters']))
	
	plot([-2, NUM_SECONDS + 2], energy, 'k--', line_width=1.2)

	xlabel('time in seconds')
	ylabel('average energy')
	title('toy example')

	gca().width = 7
	gca().height = 7
	gca().xmin = -1
	gca().xmax = NUM_SECONDS

	savefig('results/toyexample/toyexample_trace.tex')

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
