import sys
import unittest

sys.path.append('./code')

from transforms import Transform, UnivariateGaussianization
from models import MoGaussian
from numpy import abs, all

Transform.VERBOSITY = 0

class Test(unittest.TestCase):
	def test_inverse(self):
		"""
		Make sure inverse Gaussianization is inverse to Gaussianization.
		"""

		mog = MoGaussian(10)
		mog.initialize('laplace')

		# generate test data
		samples = mog.sample(100)

		ug = UnivariateGaussianization(mog)

		# reconstructed samples
		samples_ = ug.inverse(ug(samples))

		# distance between norm and reconstructed sample
		dist = abs(samples_ - samples)

		self.assertTrue(all(dist < 1E-6))

		###

		mog = MoGaussian(2)
		mog.scales[0] = 1.
		mog.scales[1] = 2.
		mog.means[0] = -4.
		mog.means[1] = 3.

		# generate test data
		samples = mog.sample(100)

		ug = UnivariateGaussianization(mog)

		# reconstructed samples
		samples_ = ug.inverse(ug(samples))

		# distance between norm and reconstructed sample
		dist = abs(samples_ - samples)

		self.assertTrue(all(dist < 1E-6))



	def test_logjacobian(self):
		"""
		Test log-Jacobian.
		"""

		# test one-dimensional Gaussian
		mog = MoGaussian(10)
		mog.initialize('laplace')

		# standard normal distribution
		gauss = MoGaussian(1)
		gauss.means[0] = 0.
		gauss.scales[0] = 1.

		# generate test data
		samples = mog.sample(100)

		ug = UnivariateGaussianization(mog)

		# after Gaussianization, samples should be Gaussian distributed
		loglik_mog = mog.loglikelihood(samples)
		loglik_gauss = gauss.loglikelihood(ug(samples)) + ug.logjacobian(samples)

		dist = abs(loglik_mog - loglik_gauss)

		self.assertTrue(all(dist < 1E-6))

		###

		# test one-dimensional Gaussian
		mog = MoGaussian(2)
		mog.scales[0] = 1.
		mog.scales[1] = 2.
		mog.means[0] = -4.
		mog.means[1] = 3.

		# standard normal distribution
		gauss = MoGaussian(1)
		gauss.means[0] = 0.
		gauss.scales[0] = 1.

		# generate test data
		samples = mog.sample(100)

		ug = UnivariateGaussianization(mog)

		# after Gaussianization, samples should be Gaussian distributed
		loglik_mog = mog.loglikelihood(samples)
		loglik_gauss = gauss.loglikelihood(ug(samples)) + ug.logjacobian(samples)

		dist = abs(loglik_mog - loglik_gauss)

		self.assertTrue(all(dist < 1E-6))



if __name__ == '__main__':
	unittest.main()
