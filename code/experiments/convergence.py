"""
Check for convergence of AIS estimate.
"""

import sys

sys.path.append('./code')

from numpy import *
from numpy import min
from numpy.random import permutation
from pgf import *
from tools import Experiment, logmeanexp
from glob import glob

def main(argv):
	base_path = argv[1]

	ais_weights = []
	indices = []

	# load results
	for path in glob(base_path[:-4] + '[0-9]*[0-9].xpck'):
		results = Experiment(path)

		indices.append(results['indices'])
		ais_weights.append(results['ais_weights'])

	# make sure each data point is used only once
	indices = hstack(indices).tolist()
	indices, idx = unique(indices, return_index=True)
	ais_weights = hstack(ais_weights)[:, idx]

	num_samples = []
	num_samples = 2**arange(0, ceil(log2(ais_weights.shape[0])) + 1)
	num_samples[-1] = max([num_samples[-1], ais_weights.shape[0]])
	estimates = []
	num_repetitions = ceil(1024. / num_samples)

	for k in arange(len(num_samples)):
		estimates_ = []

		for _ in arange(num_repetitions[k]):
			# pick samples at random
			idx = permutation(ais_weights.shape[0])[:num_samples[k]]

			# estimate log-likelihood using num_samples[k] samples
			estimates_.append(mean(logmeanexp(ais_weights[idx, :], 0)))
		estimates.append(mean(estimates_))

	subplot(0, 1)
	semilogx(num_samples, estimates, '.-')
	xlabel('number of AIS samples')
	ylabel('estimated log-likelihood')
	gca().ymin = 1.24
	gca().ymax = 1.41
	gca().width = 5
	gca().height = 5
	subplot(0, 0)
	gca().width = 5
	gca().height = 5
	semilogx(num_samples, estimates, '.-')
	xlabel('number of AIS samples')
	ylabel('estimated log-likelihood')
	draw()

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
