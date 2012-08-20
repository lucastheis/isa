"""
Evaluate overcomplete linear models.
"""

import sys

sys.path.append('./code')

from tools import Experiment, preprocess, logmeanexp
from numpy import load, mean, log, min, max, std, sqrt, all, isnan
from isa import ISA, GSM

NUM_AIS_SAMPLES = 256
NUM_AIS_STEPS = 1000

def main(argv):
	if len(argv) < 2:
		print 'Usage:', argv[0], '<experiment>', '[data_points]'
		return 0

	experiment = Experiment()

	# range of data points evaluated
	if len(argv) < 3:
		fr, to = 0, 1000
	else:
		if '-' in argv[2]:
			fr, to = argv[2].split('-')
			fr, to = int(fr), int(to)
		else:
			fr, to = 0, int(argv[2])

	indices = range(fr, to)

	# load experiment with trained model
	results = Experiment(argv[1])

	# generate test data
	data = load('data/vanhateren.{0}.0.npz'.format(results['parameters'][0]))['data']
	data = preprocess(data, shuffle=False)

	params = results['model'].model[1].model.default_parameters()
	params['ais']['num_samples'] = NUM_AIS_SAMPLES
	params['ais']['num_iter'] = NUM_AIS_STEPS

	model = results['model']



	# compute importance weights estimating likelihoods
	ais_weights = model.loglikelihood(data[:, indices],
		parameters=params, return_all=True)

	# compute average log-likelihood in [bit/pixel]
	loglik = logmeanexp(ais_weights, 0) / log(2.) / data.shape[0]
	loglik = loglik[:, -isnan(loglik)] # TODO: resolve NANs

	sem = std(loglik, ddof=1) / sqrt(loglik.size)

	loglik = mean(loglik)

	# store results
	experiment['indices'] = indices
	experiment['ais_weights'] = ais_weights
	experiment['loglik'] = loglik
	experiment['sem'] = sem
	experiment['fixed'] = True
	experiment.save(argv[1][:-4] + '{0}-{1}.xpck'.format(fr, to))

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
