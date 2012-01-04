import sys
import unittest

sys.path.append('./code')

from models import ISA, GSM
from transforms import Transform, SubspaceGaussianization
from numpy import sqrt, sum, square, all, max, vstack

Transform.VERBOSITY = 0

class Tests(unittest.TestCase):
	def test_inverse(self):
		"""
		Make sure inverse Gaussianization is inverse to Gaussianization.
		"""

		# complete model
		isa = ISA(20, 20, 2)

		# generate sample data
		samples = isa.sample(100)

		sg = SubspaceGaussianization(isa)

		# apply what should be the identity
		samples_rec = sg.inverse(sg(samples))

		# distance between samples and reconstructed samples
		dist = sqrt(sum(square(samples - samples_rec), 0))

		self.assertTrue(all(dist < 1E-6))

		###

		# overcomplete model
		isa = ISA(3, 6, 3)

		# generate sample data
		samples = isa.sample(100)
		samples = vstack([samples, isa.sample_nullspace(samples)])

		sg = SubspaceGaussianization(isa)

		# apply what should be the identity
		samples_rec = sg.inverse(sg(samples))

		# distance between samples and reconstructed samples
		dist = sqrt(sum(square(samples - samples_rec), 0))

		self.assertTrue(all(dist < 1E-6))



	def test_logjacobian(self):
		isa = ISA(4, 4, 2)

		# standard normal distribution
		gauss = GSM(4, 1)
		gauss.scales[0] = 1.

		# generate test data
		samples = isa.sample(100)

		sg = SubspaceGaussianization(isa)

		# after Gaussianization, samples should be Gaussian distributed
		loglik_isa = isa.loglikelihood(samples)
		loglik_gauss = gauss.loglikelihood(sg(samples)) + sg.logjacobian(samples)

		dist = abs(loglik_isa - loglik_gauss)

		self.assertTrue(all(dist < 1E-6))

		###

		# test ICA
		isa = ISA(3, 3, 1)

		# standard normal distribution
		gauss = GSM(3, 1)
		gauss.scales[0] = 1.

		# generate test data
		samples = isa.sample(100)

		sg = SubspaceGaussianization(isa)

		# after Gaussianization, samples should be Gaussian distributed
		loglik_isa = isa.loglikelihood(samples)
		loglik_gauss = gauss.loglikelihood(sg(samples)) + sg.logjacobian(samples)

		dist = abs(loglik_isa - loglik_gauss)

		self.assertTrue(all(dist < 1E-6))




if __name__ == '__main__':
	unittest.main()
