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

def gaussll(data):
	data = data - mean(data)

	S = inv(cov(data))

	return mean(-sum(multiply(data, dot(S, data)), 0) / 2.) \
		+ log(det(S)) / 2. \
		- log(2. * pi) * data.shape[0] / 2.


def main(argv):
	data = load('data/vanhateren.8x8.1.npz')['data']
	data = preprocess(data)

	model = ISA(data.shape[0], data.shape[0], ssize=32)
	model.train(data[:, :50000])

#	patchutil.show(model.A.T.reshape(-1, 8, 8))

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
