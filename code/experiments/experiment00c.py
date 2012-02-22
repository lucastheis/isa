"""
Apply ICA to radially Gaussianized data.
"""

import sys

sys.path.append('./code')

from models import GSM, Distribution, ISA
from transforms import RadialGaussianization
from tools import preprocess, patchutil, Experiment
from numpy import *
from matplotlib.pyplot import figure

def main(argv):
	experiment = Experiment()

	# load natural image patches
	data = load('./data/vanhateren.8x8.1.npz')['data']

	data = preprocess(data, noise_level=32)

	Distribution.VERBOSITY = 3

	gsm = GSM(data.shape[0])
	gsm.train(data, max_iter=5)

	print 'Gaussianizing...',
	data = RadialGaussianization(gsm).apply(data[:, :1000])
	print '[OK]'

	Distribution.VERBOSITY = 2

	ica = ISA(data.shape[0])
	ica.train(data, max_iter=100)

	experiment['ica'] = ica
	experiment['gsm'] = gsm
	experiment.save('./results/experiment00c/experiment00c.{0}.{1}.xpck')

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
