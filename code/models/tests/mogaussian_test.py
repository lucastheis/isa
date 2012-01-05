import sys
import unittest

sys.path.append('./code')

from models import MoGaussian, GSM, Distribution
from numpy import seterr, all, abs, log, square, pi
from numpy.random import randn
from matplotlib.pyplot import hist

Distribution.VERBOSITY = 0

class Tests(unittest.TestCase):
	def test_energy_gradient(self):
		"""
		Tests whether the energy gradient is similar to a numerical gradient.
		"""

		step_size = 1E-4

		model = MoGaussian(num_components=4)

		# samples and true gradient
		X = model.sample(100)
		G = model.energy_gradient(X)

		# numerical gradient
		G_ = model.energy(X + step_size) - model.energy(X - step_size)
		G_ = G_ / (2. * step_size)

		# test consistency of energy and gradient
		self.assertTrue(all(abs(G - G_) < 1E-5))



	def test_loglikelihood(self):
		# Gaussian
		gaussian = MoGaussian(4)
		gaussian.means[:] = 0.
		gaussian.scales[:] = 2.

		samples = gaussian.sample(100)

		# should be Gaussian log-likelihood
		loglik = gaussian.loglikelihood(samples)

		# Gaussian log-likelihood
		loglik_ = -square(samples) / 8. - log(8. * pi) / 2.

		self.assertTrue(all(abs(loglik - loglik_) < 1E-8))



	def test_train(self):
		seterr(divide='raise', over='raise', invalid='raise')

		gsm = GSM(1, 10)
		gsm.initialize(method='cauchy')


		samples = gsm.sample(5000)

		mog = MoGaussian(num_components=10)
		mog.initialize(method='laplace')
		mog.train(samples, 100)



	def test_sample_posterior(self):
		mog = MoGaussian()
		samples = mog.sample(1000)

		means, scales = mog.sample_posterior(samples)

		self.assertTrue(samples.ndim == 2)
		self.assertTrue(samples.shape[0] == 1)
		self.assertTrue(means.ndim == 2)
		self.assertTrue(means.shape[0] == 1)
		self.assertTrue(scales.ndim == 2)
		self.assertTrue(scales.shape[0] == 1)



if __name__ == '__main__':
	unittest.main()
