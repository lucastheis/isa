import sys
import unittest 

sys.path.append('./code')

from models import GSM
from models.utils import logmeanexp
from numpy import zeros, all, abs, array, square, log, pi, sum, mean
from numpy.random import randn, rand

class Tests(unittest.TestCase):
	def test_energy(self):
		step_size = 1E-4

		model = GSM(3, num_scales=7)

		# samples and true gradient
		X = model.sample(100)
		G = model.energy_gradient(X)

		# numerical gradient
		N = zeros(G.shape)
		for i in range(N.shape[0]):
			d = zeros(X.shape)
			d[i] = step_size
			N[i] = (model.energy(X + d) - model.energy(X - d)) / (2. * step_size)

		# test consistency of energy and gradient
		self.assertTrue(all(abs(G - N) < 1E-5))



	def test_loglikelihood(self):
		for dim in [1, 2, 3, 4, 5]:
			for num_scales in [1, 2, 3, 4, 5]:
				# create Gaussian scale mixture
				model = GSM(dim, num_scales=num_scales)
				scales = model.scales.reshape(-1, 1)

				# create random data
				data = randn(model.dim, 100)

				# evaluate likelihood
				ll = logmeanexp(
					-0.5 * sum(square(data), 0) / square(scales)
					- model.dim * log(scales)
					- model.dim / 2. * log(2. * pi), 0)

				self.assertTrue(all(abs(ll - model.loglikelihood(data)) < 1E-6))

				# random scales
				scales = rand(num_scales, 1) + 0.5
				model.scales[:] = scales.flatten()

				# sample data from model
				data = model.sample(100)

				# evaluate likelihood
				ll = logmeanexp(
					-0.5 * sum(square(data), 0) / square(scales)
					- model.dim * log(scales)
					- model.dim / 2. * log(2. * pi), 0)

				self.assertTrue(all(abs(ll - model.loglikelihood(data)) < 1E-6))



if __name__ == '__main__':
	unittest.main()
