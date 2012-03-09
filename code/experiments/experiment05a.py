"""
Train hierarchical ISA on CIFAR-10.
"""

import sys

sys.path.append('./code')

from tools import Experiment, cifar
from models import GSM, ISA, StackedModel
from transforms import LinearTransform, WhiteningTransform
from transforms import SubspaceGaussianization, RadialGaussianization

from numpy import dot, min, max
from numpy.linalg import pinv



def reconstruct(images, wt, rg):
	images = dot(pinv(wt.A[:1024]), rg.inverse(images))
	images = images.T.reshape(-1, 32, 32, 3)
	return images



def main(argv):
	experiment = Experiment()

	data = cifar.load([1, 2, 3, 4, 5])[0]
	data = cifar.preprocess(data)

	# apply PCA whitening and reduce dimensionality
	wt = WhiteningTransform(data, symmetric=False)
	data = wt(data)[:1024]

	# train Gaussian scale mixture
	gsm = GSM(data.shape[0], 20)
	gsm.train(data, max_iter=200, tol=1e-7)

	# radially Gaussianize data
	rg = RadialGaussianization(gsm)
	data = rg(data)

	transforms = [wt, rg]

	for _ in range(2):
		isa = ISA(data.shape[0])
		isa.initialize(method='laplace')

		# initialize, train and fine-tune ISA
		isa.train(data,
			max_iter=50, train_prior=False, train_subspaces=False, method='sgd')
		isa.train(data,
			max_iter=100, train_prior=True, train_subspaces=True, method='sgd')
		isa.train(data,
			max_iter=20, train_prior=True, train_subspaces=True, method='lbfgs')

		# subspace Gaussianize data
		sg = SubspaceGaussianization(isa)
		data = sg(data)

		transforms.append(sg)

		experiment['transforms'] = transforms
		experiment['model'] = isa
		experiment.save('results/experiment05a/experiment05a.{0}.{1}.xpck')

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
