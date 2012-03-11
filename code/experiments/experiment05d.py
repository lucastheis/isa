"""
Visualize filters learned on CIFAR-10.
"""

import sys

sys.path.append('./code')

from tools import Experiment, cifar
from numpy import min, max, dot, sqrt, sum, square, argsort, sort, round
from numpy.linalg import pinv
from matplotlib.pyplot import clf, imshow, show



def reconstruct(images, wt, rg=None):
	if rg is None:
		return dot(pinv(wt.A[:1024]), images)
	return dot(pinv(wt.A[:1024]), rg.inverse(images))



def main(argv):
	if len(argv) < 2:
		print 'Usage:', argv[0], '<experiment>'
		return 0

	results = Experiment(argv[1])

	wt, rg, sg = results['transforms'][:3]

	A = reconstruct(sg.isa.A, wt)

	# symmetrically whiten filters
	wt, = Experiment('results/experiment05a/wt.xpck')['wt']
	A = wt(A)
	
	# sort and reshape filters
	n = sqrt(sum(square(A), 0))
	i = argsort(n)
	A = A.T.reshape(-1, 32, 32, 3)
	A = A[i]

	# normalize filters

	for i in range(A.shape[0]):
		a = min(A[i])
		b = max(A[i])
		f = (A[i] - a) / (b - a)
		clf()
		print round(sort(sg.isa.subspaces[i].scales)[::-1], 2)
		imshow(f, interpolation='nearest')
		show()

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
