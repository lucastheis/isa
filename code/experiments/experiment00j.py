"""
Compute symmetric whitening matrix from CIFAR-10.
"""

import sys

sys.path.append('./code')

from tools import Experiment, cifar
from models import GSM, ISA, StackedModel
from transforms import LinearTransform, WhiteningTransform
from transforms import SubspaceGaussianization, RadialGaussianization

from numpy import dot, min, max
from numpy.linalg import pinv



def main(argv):
	experiment = Experiment()

	data = cifar.load([1, 2, 3, 4, 5])[0]
	data = cifar.preprocess(data)

	# apply PCA whitening and reduce dimensionality
	wt = WhiteningTransform(data, symmetric=True)

	experiment['wt'] = [wt]
	experiment.save('results/experiment05a/wt.xpck')

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
