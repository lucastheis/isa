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
from tools import contours, mapp, preprocess
from pgf import *
from pdb import set_trace
from time import time
from tools import Experiment
from copy import deepcopy

NUM_SAMPLES = 200
NUM_SECONDS = 90
NUM_STEPS_MULTIPLIER = 5

EXPERIMENT_PATH = 'results/vanhateren/vanhateren.7.08042012.150147.xpck'

# transition operator parameters
sampling_methods = [
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 2,
			'lf_num_steps': 2,
			'lf_step_size': 0.025,
			'lf_randomness': 0.1,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 2,
			'lf_num_steps': 2,
			'lf_step_size': 0.05,
			'lf_randomness': 0.1,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 2,
			'lf_num_steps': 2,
			'lf_step_size': 0.1,
			'lf_randomness': 0.1,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 2,
			'lf_num_steps': 2,
			'lf_step_size': 0.15,
			'lf_randomness': 0.1,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 2,
			'lf_num_steps': 5,
			'lf_step_size': 0.05,
			'lf_randomness': 0.1,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 2,
			'lf_num_steps': 5,
			'lf_step_size': 0.75,
			'lf_randomness': 0.01,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 2,
			'lf_num_steps': 5,
			'lf_step_size': 0.1,
			'lf_randomness': 0.01,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 2,
			'lf_num_steps': 5,
			'lf_step_size': 0.125,
			'lf_randomness': 0.01,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 2,
			'lf_num_steps': 5,
			'lf_step_size': 0.75,
			'lf_randomness': 0.05,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 2,
			'lf_num_steps': 5,
			'lf_step_size': 0.1,
			'lf_randomness': 0.05,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 2,
			'lf_num_steps': 5,
			'lf_step_size': 0.125,
			'lf_randomness': 0.05,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 2,
			'lf_num_steps': 5,
			'lf_step_size': 0.75,
			'lf_randomness': 0.1,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 2,
			'lf_num_steps': 5,
			'lf_step_size': 0.1,
			'lf_randomness': 0.1,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 2,
			'lf_num_steps': 5,
			'lf_step_size': 0.125,
			'lf_randomness': 0.1,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 2,
			'lf_num_steps': 10,
			'lf_step_size': 0.025,
			'lf_randomness': 0.05,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 2,
			'lf_num_steps': 10,
			'lf_step_size': 0.05,
			'lf_randomness': 0.05,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 2,
			'lf_num_steps': 10,
			'lf_step_size': 0.075,
			'lf_randomness': 0.05,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 2,
			'lf_num_steps': 10,
			'lf_step_size': 0.025,
			'lf_randomness': 0.01,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 2,
			'lf_num_steps': 10,
			'lf_step_size': 0.05,
			'lf_randomness': 0.01,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 2,
			'lf_num_steps': 10,
			'lf_step_size': 0.075,
			'lf_randomness': 0.01,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 2,
			'lf_num_steps': 10,
			'lf_step_size': 0.025,
			'lf_randomness': 0.1,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 2,
			'lf_num_steps': 10,
			'lf_step_size': 0.05,
			'lf_randomness': 0.1,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 2,
			'lf_num_steps': 10,
			'lf_step_size': 0.075,
			'lf_randomness': 0.1,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 2,
			'lf_num_steps': 10,
			'lf_step_size': 0.1,
			'lf_randomness': 0.1,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 2,
			'lf_num_steps': 10,
			'lf_step_size': 0.125,
			'lf_randomness': 0.1,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 2,
			'lf_num_steps': 20,
			'lf_step_size': 0.05,
			'lf_randomness': 0.1,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 2,
			'lf_num_steps': 20,
			'lf_step_size': 0.075,
			'lf_randomness': 0.1,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 2,
			'lf_num_steps': 20,
			'lf_step_size': 0.075,
			'lf_randomness': 0.05,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 2,
			'lf_num_steps': 20,
			'lf_step_size': 0.075,
			'lf_randomness': 0.01,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 2,
			'lf_num_steps': 20,
			'lf_step_size': 0.1,
			'lf_randomness': 0.1,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 2,
			'lf_num_steps': 30,
			'lf_step_size': 0.025,
			'lf_randomness': 0.1,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 2,
			'lf_num_steps': 30,
			'lf_step_size': 0.05,
			'lf_randomness': 0.1,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 2,
			'lf_num_steps': 30,
			'lf_step_size': 0.075,
			'lf_randomness': 0.1,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 2,
			'lf_num_steps': 30,
			'lf_step_size': 1.0,
			'lf_randomness': 0.1,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'hmc', 
		'parameters': {
			'num_steps': 2,
			'lf_num_steps': 50,
			'lf_step_size': 0.05,
			'lf_randomness': 0.1,
		},
		'color': RGB(0.6, 0.6, 0.6),
	},
	{
		'method': 'mala', 
		'parameters': {
			'num_steps': 5,
			'step_width': 0.05,
		},
		'color': RGB(0.8, 0.8, 0.8),
	},
	{
		'method': 'mala', 
		'parameters': {
			'num_steps': 5,
			'step_width': 0.075,
		},
		'color': RGB(0.8, 0.8, 0.8),
	},
	{
		'method': 'mala', 
		'parameters': {
			'num_steps': 5,
			'step_width': 0.0875,
		},
		'color': RGB(0.8, 0.8, 0.8),
	},
	{
		'method': 'mala', 
		'parameters': {
			'num_steps': 5,
			'step_width': 0.1,
		},
		'color': RGB(0.8, 0.8, 0.8),
	},
	{
		'method': 'mala', 
		'parameters': {
			'num_steps': 5,
			'step_width': 0.1125,
		},
		'color': RGB(0.8, 0.8, 0.8),
	},
	{
		'method': 'mala', 
		'parameters': {
			'num_steps': 5,
			'step_width': 0.125,
		},
		'color': RGB(0.8, 0.8, 0.8),
	},
	{
		'method': 'mala', 
		'parameters': {
			'num_steps': 5,
			'step_width': 0.15,
		},
		'color': RGB(0.8, 0.8, 0.8),
	},
	{
		'method': 'mala', 
		'parameters': {
			'num_steps': 5,
			'step_width': 0.175,
		},
		'color': RGB(0.8, 0.8, 0.8),
	},
	{
		'method': 'mala', 
		'parameters': {
			'num_steps': 5,
			'step_width': 0.2,
		},
		'color': RGB(0.8, 0.8, 0.8),
	},
	{
		'method': 'gibbs',
		'parameters': {
			'num_steps': 5,
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

	X = data

	for method in sampling_methods:
		# disable output and parallelization
		Distribution.VERBOSITY = 0
		mapp.max_processes = 1

		# measure time required by transition operator
		start = time()

		# initial hidden states
		Y = dot(pinv(ica.A), X)

		# increase number of steps to reduce overhead
		ica.sample_posterior(X, method=(method['method'], dict(method['parameters'],
			Y=Y, num_steps=method['parameters']['num_steps'] * NUM_STEPS_MULTIPLIER)))

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

	xlabel('time in seconds')
	ylabel('average energy')
	title('van Hateren')

	gca().width = 7
	gca().height = 7
	gca().xmin = -1
	gca().xmax = NUM_SECONDS

	savefig('results/vanhateren/vanhateren_trace.tex')

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
