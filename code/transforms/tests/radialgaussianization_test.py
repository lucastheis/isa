import sys
import unittest 

sys.path.append('./code')

from transforms import RadialGaussianization
from models import GSM
from numpy import all, sqrt, sum, square

class Tests(unittest.TestCase):
	def test_inverse(self):
		"""
		Make sure inverse Gaussianization is inverse to Gaussianization.
		"""

		gsm = GSM(3, 10)
		gsm.initialize('cauchy')

		# generate test data
		samples = gsm.sample(100)

		rg = RadialGaussianization(gsm)

		# reconstructed samples
		samples_ = rg.inverse(rg(samples))

		# distance between norm and reconstructed norm
		dist = abs(sqrt(sum(square(samples_))) - sqrt(sum(square(samples))))

		self.assertTrue(all(dist < 1E-8))



	def test_logjacobian(self):
		"""
		Test log-Jacobian.
		"""

		gsm = GSM(3, 10)
		gsm.initialize('cauchy')

		# standard normal distribution
		gauss = GSM(3, 1)
		gauss.scales[0] = 1.

		# generate test data
		samples = gsm.sample(100)

		rg = RadialGaussianization(gsm)

		# after Gaussianization, samples should be Gaussian distributed
		loglik_gsm = gsm.loglikelihood(samples)
		loglik_gauss = gauss.loglikelihood(rg(samples)) + rg.logjacobian(samples)

		dist = abs(loglik_gsm - loglik_gauss)

		self.assertTrue(all(dist < 1E-8))



if __name__ == '__main__':
	unittest.main()
