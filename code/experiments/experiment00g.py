import os
import sys

sys.path.append('./code')

from numpy import *
from pgf import *
from models import GSM
from tools import Experiment, mapp

filepath = 'results/experiment00f/'

def main(argv):
	entropy = {}
	logloss = {}

	for filename in os.listdir(filepath):
		if filename.endswith('.xpck'):
			experiment = Experiment(os.path.join(filepath, filename))

			if experiment['ica'].num_hiddens not in entropy:
				entropy[experiment['ica'].num_hiddens] = []
				logloss[experiment['ica'].num_hiddens] = []

			entropy[experiment['ica'].num_hiddens].append(experiment['entropy'])
			logloss[experiment['ica'].num_hiddens].append(experiment['logloss'])

	keys = sort(entropy.keys())
	entropy_err = [sem(entropy[key]) for key in keys]
	entropy = [mean(entropy[key]) for key in keys]
	logloss_err = [sem(logloss[key]) for key in keys]
	logloss = [mean(logloss[key]) for key in keys]

	plot(keys, entropy, yerr=entropy_err)
	plot(keys, logloss, yerr=logloss_err)
	draw()

	return 0



def sem(values):
	return std(values) / sqrt(len(values))



if __name__ == '__main__':
	sys.exit(main(sys.argv))
