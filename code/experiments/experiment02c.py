"""
Plot results for ICA fitted to GSM.
"""

import os
import sys

sys.path.append('./code')

from tools import Experiment
from numpy import *
from numpy import min, max
from pgf import *

filepath = './results/experiment02b/'
 
def main(argv):
	entropy = {}
	logloss = {}

	for filename in os.listdir(filepath):
		if filename.endswith('.xpck'):
			experiment = Experiment(os.path.join(filepath + filename))

			if experiment['ica'].num_hiddens not in entropy:
				entropy[experiment['ica'].num_hiddens] = []
				logloss[experiment['ica'].num_hiddens] = []

			if experiment['logloss'] < 3.:
				entropy[experiment['ica'].num_hiddens].append(experiment['entropy'])
				logloss[experiment['ica'].num_hiddens].append(experiment['logloss'])

	print [len(entropy[k]) for k in entropy.keys()]

	keys = sort(entropy.keys())

	for k in logloss.keys():
		print str(k) + ': ',
		for l in sort(logloss[k]):
			print '{0:8.4f}'.format(l), 
		print

	entropy_err = [sem(entropy[key]) * 2. for key in keys]
	entropy = [mean(entropy[key]) for key in keys]

	logloss_err = [sem(logloss[key]) * 2. for key in keys]
	logloss_min = [min(logloss[key]) for key in keys]
	logloss_med = [median(logloss[key]) for key in keys]
	logloss = [mean(logloss[key]) for key in keys]

	print logloss_min

	plot(keys, entropy, 'k--')
	plot(keys, logloss, color=RGB(0.7, 0.7, 0.7), yerr=logloss_err, line_width=2., error_width=2., error_marker='none')
	plot(keys, logloss_med, 'b')
	plot(keys, logloss_min, 'r-.')
	ylabel('log-loss $\pm$ $2 \cdot$ SEM')
	xlabel('number of hidden units')
	grid()
	legend('GSM', 'ICA (mean)', 'ICA (median)', 'ICA (minimum)')
	draw()

	return 0



def sem(values):
	return std(values, ddof=1) / sqrt(len(values))



if __name__ == '__main__':
	sys.exit(main(sys.argv))
