"""
Show filters of a trained model.
"""

import sys

sys.path.append('./code')

from tools import Experiment, patchutil
from matplotlib.pyplot import *
from numpy import sqrt, dot, square, sum, argsort, sort

def main(argv):
	e = Experiment(argv[1])

	print e['parameters']

	if 'transforms' in e.results:
		m = e['model'][1].model

		dct, wt = e['transforms']

		A = dot(dct.A[1:].T, wt.inverse(m[1].model.A))
		A_white = dot(dct.A[1:].T, m[1].model.A)

		print m.A.shape
		if '_noise' in m[1].model.__dict__:
			if m.noise:
				A = A[:, m.num_visibles:]
				L = A_white[:, :m.num_visibles]
				A_white = A_white[:, m.num_visibles:]

		if '_noise' in m.__dict__:
			if m.noise:
				figure()
				patchutil.show(L.T.reshape(-1, patch_size, patch_size))

	else:
		m = e['model']
		A = m.A
		A_white = m.A

	if m.subspaces[0].dim == 1:
		n = sqrt(sum(square(A_white), 0))
		i = argsort(n)[::-1]

		A = A[:, i]
		A_white = A_white[:, i]

	A = A / sqrt(sum(square(A), 0))

	patch_size = int(sqrt(A.shape[0]) + 0.5)

	figure()
	patchutil.show(A.T.reshape(-1, patch_size, patch_size), num_rows=patch_size)

	figure()
	patchutil.show(A_white.T.reshape(-1, patch_size, patch_size))

	figure()
	plot(sort(sqrt(sum(square(A_white), 0)))[::-1])
	axis([0, A.shape[1], 0, 1.2])
	xlabel('component')
	ylabel('feature norm / component standard deviation')

	show()

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
