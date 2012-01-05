import sys
import unittest

sys.path.append('./code')

from models import MoGaussian, GSM, Distribution
from numpy import seterr, all, abs
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



	def test_train(self):
		seterr(divide='raise', over='raise', invalid='raise')

		gsm = GSM(1, 10)
		gsm.initialize('cauchy')


		samples = gsm.sample(5000)

		mog = MoGaussian(num_components=10)
		mog.train(samples, 100)



if __name__ == '__main__':
	unittest.main()
