"""
Visualize filters learned on CIFAR-10.
"""

import sys

sys.path.append('./code')

from tools import Experiment, cifar
from numpy import min, max, dot, sqrt, sum, square, argsort, sort, round
from numpy.linalg import pinv
from tools import imsave, imformat, stitch



def reconstruct(images, wt, rg=None):
	if rg is None:
		return dot(pinv(wt.A[:1024]), images)
	return dot(pinv(wt.A[:1024]), rg.inverse(images))



def main(argv):
	if len(argv) < 2:
		print 'Usage:', argv[0], '<experiment>'
		return 0

	results = Experiment(argv[1])

	wt, = results['transforms'][:1]
	isa = results['model']

	A = reconstruct(isa.A, wt)

	# symmetrically whiten filters
#	wt, = Experiment('results/experiment05a/wt.xpck')['wt']
#	A = wt(A)
	
	# sort and reshape filters
	n = sqrt(sum(square(A), 0))
	i = argsort(n)[::-1]
	A = A.T.reshape(-1, 32, 32, 3)
	A = A[i]
	A = A

	imsave('/kyb/agmb/lucas/Projects/isa/results/experiment05a/cifar.png', stitch(imformat(A)))

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
