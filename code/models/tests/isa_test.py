import sys
import unittest 

sys.path.append('./code')

from models import ISA
from numpy import zeros, all, abs

class Tests(unittest.TestCase):
	def test_prior_energy(self):
		step_size = 1E-4

		model = ISA(3, 7)

		# samples and true gradient
		X = model.sample_prior(100)
		G = model.prior_energy_gradient(X)

		# numerical gradient
		N = zeros(G.shape)
		for i in range(N.shape[0]):
			d = zeros(X.shape)
			d[i] = step_size
			N[i] = (model.prior_energy(X + d) - model.prior_energy(X - d)) / (2. * step_size)

		# test consistency of energy and gradient
		self.assertTrue(all(abs(G - N) < 1E-5))



if __name__ == '__main__':
	unittest.main()

