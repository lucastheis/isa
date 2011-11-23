"""
"""

import sys

sys.path.append('./code')

from numpy import *
from numpy.linalg import *
from models import ISA
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

	model = ISA(data.shape[0], data.shape[0], ssize=2)
	model.initialize(data[:, :20000])

	model.train(data[:, :20000], max_iter=100)

	model.A = model.A / sqrt(sum(square(model.A), 0)).reshape(1, -1)

	patchutil.show(model.A.T.reshape(-1, 8, 8))

	raw_input()

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
