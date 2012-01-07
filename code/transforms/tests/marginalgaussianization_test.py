import sys
import unittest

sys.path.append('./code')

from models import ICA, GSM
from transforms import Transform, MarginalGaussianization
from numpy import sqrt, sum, square, all, max, vstack

Transform.VERBOSITY = 0

class Tests(unittest.TestCase):
	def test_inverse(self):
		"""
		Make sure inverse Gaussianization is inverse to Gaussianization.
		"""

		# complete model
		ica = ICA(11)

		# generate sample data
		samples = ica.sample(100)

		mg = MarginalGaussianization(ica)

		# apply what should be the identity
		samples_rec = mg.inverse(mg(samples))

		# distance between samples and reconstructed samples
		dist = sqrt(sum(square(samples - samples_rec), 0))

		self.assertTrue(all(dist < 1E-6))



	def test_logjacobian(self):
		ica = ICA(4)

		# standard normal distribution
		gauss = GSM(4, 1)
		gauss.scales[0] = 1.

		# generate test data
		samples = ica.sample(100)

		mg = MarginalGaussianization(ica)

		# after Gaussianization, samples should be Gaussian distributed
		loglik_ica = ica.loglikelihood(samples)
		loglik_gauss = gauss.loglikelihood(mg(samples)) + mg.logjacobian(samples)

		dist = abs(loglik_ica - loglik_gauss)

		self.assertTrue(all(dist < 1E-6))



if __name__ == '__main__':
	unittest.main()
