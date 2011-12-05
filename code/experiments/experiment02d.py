"""
Generate contour plots for best ICA model and GSM.
"""

import os
import sys

sys.path.append('./code')

from tools import Experiment
from numpy import *
from numpy import min, max
from matplotlib.pyplot import *
from tools import contours

filepath = './results/experiment02b/'
 
def main(argv):
	entropy = {}
	logloss = {}

	results = Experiment('results/experiment02a/experiment02a.xpck')

	# load Gaussian scale mixture
	gsm = results['gsm']

	# load best ICA model
	ll = inf

	for filename in os.listdir(filepath):
		if filename.endswith('.xpck'):
			experiment = Experiment(os.path.join(filepath + filename))

			if experiment['logloss'] < ll:
				ll = experiment['logloss']
				ica = experiment['ica']
	

	data = gsm.sample(5000000)
	samples = ica.sample(5000000)

	levels = arange(0.025, 1., 0.04)

	# contour plots
	figure()
	contours(data[:2], 100, levels, colors='k')
	axis('equal')
	ax = axis()
	title('joint distribution (GSM)')
	savefig('results/experiment02d/j_gsm.png')

	figure()
	contours(samples[:2], 100, levels, colors='k')
	axis('equal')
	axis(ax)
	title('joint distribution (ICA)')
	savefig('results/experiment02d/j_ica.png')

	# marginal distributions
	figure()
	h, x = histogram(data[1], 200, normed=True)
	plot((x[1:] + x[:-1]) / 2., h, 'b')
	h, x = histogram(samples[1], 200, normed=True)
	plot((x[1:] + x[:-1]) / 2., h, 'r')
	title('marginal distribution')
	legend(['GSM', 'ISA'])
	savefig('results/experiment02d/marginal.png')

	raw_input()

	return 0


if __name__ == '__main__':
	sys.exit(main(sys.argv))
