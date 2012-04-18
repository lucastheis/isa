"""
Measure sampling performance.
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
		'num_steps': 5,
	},
	'mala': {
		'num_steps': 1,
		'step_width': 0.05,
	},
	'hmc': {
		'num_steps': 1,
		'lf_step_size': 0.01,
		'lf_num_steps': 1,#80,
		'lf_randomness': 0.,#0.1,
	},
}

# number of transition operator applications
num_steps = {
	'gibbs': 40,
	'mala': 20,
	'hmc': 20,
}

legend_entries = {
	'gibbs': 'Gibbs',
	'mala': 'MALA',
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

	if False:
		isa = ISA(1, 3)
		isa.initialize(method='student')
		isa.A[:] = 1.
		isa.A += randn(3) / 5.
	else:
		results = Experiment('results/experiment01a/experiment01a.27012012.204502.xpck')
		isa = results['model'][1].model



	## TIME MEASUREMENTS

	# generate true hidden and visible states
	Y = isa.sample_prior(4)
	X = dot(isa.A, Y)

	# energy sampling methods are expected to converge to
	energy_exp = mean(isa.prior_energy(Y))

	times = {}

	for sampling_method in parameters:
		params = deepcopy(parameters[sampling_method])
		params['num_steps'] *= num_steps[sampling_method]

		start = time()
		 
		isa.sample_posterior(X, method=(sampling_method, params))

		# measure time per transition operator application
		times[sampling_method] = (time() - start) / num_steps[sampling_method]




	## MCMC SAMPLES

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
	for sampling_method in parameters:
		times[sampling_method] = arange(len(energies[sampling_method])) * times[sampling_method]
		plot(times[sampling_method], energies[sampling_method])
	legend(*legend_entries.values(), location='north east')

	plot([0, max(hstack(times.values()))], [energy_exp, energy_exp], 'k--')

	xlabel('time in seconds')
	ylabel('average energy')

	savefig('convergence.tex')
#	draw()

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
