"""
Compute AIS weights and samples.
"""

import sys

sys.path.append('./code')

from tools import Experiment, preprocess, logmeanexp
from numpy import load, mean, log, min, max, std, sqrt, all, isnan, vstack, any, isnan
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

	# load test data
	data = load('data/vanhateren.{0}.0.npz'.format(results['parameters'][0]))['data']
	data = preprocess(data, shuffle=False)

	params = results['model'].model[1].model.default_parameters()
	params['ais']['num_iter'] = NUM_AIS_STEPS

	model = results['model']

	# transforms
	dct = results['model'].transforms[0]
	wt = results['model'].model[1].transforms[0]

	if len(results['model'].model[1].transforms) > 1:
		rg = results['model'].model[1].transforms[1]
		data_tr = rg(wt(dct(data[:, indices])[1:]))
	else:
		rg = None
		data_tr = wt(dct(data[:, indices])[1:])



	samples = []
	ais_weights = []

	for i in range(NUM_AIS_SAMPLES):
		# compute posterior samples and importance weights
		samples_, ais_weights_ = results['model'].model[1].model.sample_posterior_ais(
			data_tr, parameters=params)

		# store samples
		samples.append(samples_)
		ais_weights.append(ais_weights_)

		print '{0}/{1}'.format(i + 1, NUM_AIS_SAMPLES)

	ais_weights = vstack(ais_weights)



	# compute average log-likelihood in [bit/pixel]
	loglik_dc = results['model'].model[0].loglikelihood(dct(data[:, indices])[:1])
	loglik = logmeanexp(ais_weights, 0)
	loglik = loglik_dc + loglik + wt.logjacobian()

	if rg is not None:
		loglik += rg.logjacobian(wt(dct(data[:, indices])[1:]))

	loglik = loglik / log(2.) / data.shape[0]
	loglik = loglik[:, -isnan(loglik)] # TODO: resolve NANs

	sem = std(loglik, ddof=1) / sqrt(loglik.size)

	loglik = mean(loglik)



	# store results
	experiment['indices'] = indices
	experiment['ais_weights'] = ais_weights
	experiment['samples'] = samples
	experiment['loglik'] = loglik
	experiment['sem'] = sem
	experiment['fixed'] = True
	experiment.save(argv[1][:-4] + 'ais_samples.{0}-{1}.xpck'.format(fr, to))

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
