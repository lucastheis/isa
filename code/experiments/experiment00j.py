"""
Test OF algorithm.
"""

import sys

sys.path.append('./code')

from models import ISA
from transforms import WhiteningTransform
from tools import preprocess
from numpy import load

def main(argv):
	# load data, log-transform and center data
	data = load('data/vanhateren.8x8.1.npz')['data']
	data = preprocess(data, noise_level=32)
	
	# create whitening transform
	wt = WhiteningTransform(data, symmetric=True)

	data = wt(data[:, :20000])

	isa = ISA(
		num_visibles=data.shape[0],
		num_hiddens=2 * data.shape[0],
		ssize=1)

	isa.train_of(data)

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
