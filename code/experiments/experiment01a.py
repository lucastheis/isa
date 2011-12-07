"""
Train model on natural images.
"""

import sys

sys.path.append('./code')

from numpy import *
from numpy.linalg import *
from models import ISA, GSM
from tools import preprocess, patchutil
from pgf import *


def main(argv):
	data = load('data/vanhateren.8x8.1.npz')['data']
	data = preprocess(data)

	model = ISA(data.shape[0], data.shape[0], ssize=1)
	model.train(data[:, :50000])
	
	print model.evaluate(data[:, 50000:100000]) / log(2.)

#	patchutil.show(model.A.T.reshape(-1, 8, 8))

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
