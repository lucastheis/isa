"""
Train ISA on CIFAR-10 using O&F's sparse coding.
"""

import sys

sys.path.append('./code')

from tools import Experiment, cifar, imsave, stitch, imformat
from models import GSM, ISA, StackedModel
from transforms import LinearTransform, WhiteningTransform
from transforms import SubspaceGaussianization, RadialGaussianization

from numpy import dot, min, max
from numpy.linalg import pinv



def reconstruct(images, wt, rg=None):
	if rg is None:
		return dot(pinv(wt.A[:1024]), images)
	return dot(pinv(wt.A[:1024]), rg.inverse(images))



def main(argv):
	experiment = Experiment(server='newton')

	data = cifar.load([1, 2, 3, 4, 5])[0]
	data = cifar.preprocess(data)

	# apply PCA whitening and reduce dimensionality
	wt = WhiteningTransform(data, symmetric=False)
	data = wt(data)[:1024]

	isa = ISA(data.shape[0])

	experiment['transforms'] = [wt]
	experiment['model'] = isa

	def callback(isa, iteration):
		A = reconstruct(isa.A, wt)
		A = A.T.reshape(-1, 32, 32, 3)
		experiment.save('results/experiment05b/experiment05b.{0}.xpck'.format(iteration))
		imsave('results/experiment05b/cifar.{0}.png'.format(iteration),
			stitch(imformat(A)))

	isa.train_of(data,
		max_iter=50,
		noise_var=0.2,
		var_goal=1.,
		beta=10.,
		step_width=0.001,
		sigma=1.0,
		callback=callback)

	experiment.save('results/experiment05b/experiment05b.{0}.{1}.xpck')

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
