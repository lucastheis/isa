"""
Learn an orthogonal set of complete ISA filters for Holly's experiments.
"""

import sys

sys.path.append('./code')

from models import ISA
from tools import preprocess
from numpy import *

def main(argv):
	# load preprocessed image patches
	data = load('data/vanhateren.{0}.holly.npz'.format(patch_size))['data']

	isa = ISA(ssize=2)
	isa.train(data[:, :20000], max_iter=100, train_prior=False, method='sgd')
	isa.train(data[:, :20000], max_iter=100, train_prior=True, method='sgd')
	
	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
