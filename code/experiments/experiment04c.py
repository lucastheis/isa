"""
Plot performance of different sampling methods.
"""

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

Distribution.VERBOSITY = 2
mapp.max_processes = 1

# transition operator parameters
parameters = {
	'gibbs': {
		'num_steps': 1,
	},
	'hmc': {
		'num_steps': 1,
		'lf_step_size': 0.01,
		'lf_num_steps': 30
	},
}

# number of transition operator applications
num_steps = {
	'gibbs': 20,
	'hmc': 20,
}

legend_entries = {
	'gibbs': 'Gibbs',
	'hmc': 'HMC',
}

def main(argv):
	seterr(over='raise', divide='raise', invalid='raise')

	## SAMPLING METHOD

	if len(argv) > 1:
		if argv[1] not in parameters:
			raise ValueError('Unknown sampling method ''{0}'''.format(argv[1]))
		sampling_method = argv[1]
	else:
		sampling_method = 'gibbs'


	## MODEL

	isa = ISA(1, 3)
	isa.initialize(method='student')
	isa.A[:] = 1.
	for gsm in isa.subspaces:
		gsm.initialize(method='student')



	## TIME MEASUREMENTS

	# generate true hidden and visible states
	Y = isa.sample_prior(10000)
	X = dot(isa.A, Y)

	# energy sampling methods are expected to converge to
	energy_exp = mean(isa.prior_energy(Y))

	times = {}

	for sampling_method in parameters:
		params = deepcopy(parameters[sampling_method])
		params['num_steps'] *= num_steps[sampling_method]

		start = time()
		 
		# sample without interruption
		isa.sample_posterior(X, method=(sampling_method, params))

		# measure time per transition operator application
		times[sampling_method] = (time() - start) / num_steps[sampling_method]



	## GENERATE MCMC SAMPLES

	energies = {}

	for sampling_method in parameters:
		print legend_entries[sampling_method]

		# initial hidden states
		Y = dot(pinv(isa.A), X)

		# average energy of hidden states
		E = [mean(isa.prior_energy(Y))]

		for _ in range(num_steps[sampling_method]):
			# advance Markov chain
			Y = isa.sample_posterior(X, 
				method=(sampling_method, dict(parameters[sampling_method], Y=Y)))
			
			E.append(mean(isa.prior_energy(Y)))

		energies[sampling_method] = hstack(E)



	## VISUALIZE

	legentries = []

	for sampling_method in parameters:
		times[sampling_method] = arange(len(energies[sampling_method])) * times[sampling_method]
		plot(times[sampling_method], energies[sampling_method])
		legentries.append(legend_entries[sampling_method])
	legend(*legentries, location='south east')

	plot([0, max(hstack(times.values()))], [energy_exp, energy_exp], 'k--')

	xlabel('time in seconds')
	ylabel('average energy')

	draw()

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
