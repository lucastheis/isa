"""
Visualize filters learned on CIFAR-10.
"""

import sys

sys.path.append('./code')

from tools import Experiment, cifar
from numpy import min, max, dot, sqrt, sum, square, argsort, sort, round
from numpy.linalg import pinv
from matplotlib.pyplot import clf, imshow, show
from tools import stitch, imformat, imsave



def reconstruct(images, wt, rg=None):
	if rg is None:
		return dot(pinv(wt.A[:1024]), images)
	return dot(pinv(wt.A[:1024]), rg.inverse(images))



def main(argv):
	if len(argv) < 2:
		print 'Usage:', argv[0], '<experiment>'
		return 0

	results = Experiment(argv[1])

	# sample from top layer
	X = results['model'].sample(100)

	# inverse hierarchical subspace Gaussianization
	for sg in results['transforms'][2:-1]:
		X = sg.inverse(X)

	# inverse radial Gaussianization and whitening
	X = reconstruct(X, *results['transforms'][:2])
	X = X.T.reshape(-1, 32, 32, 3)

	X = stitch(imformat(X), num_rows=10)

	imsave('/Users/lucas/Desktop/samples.png', X)

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
