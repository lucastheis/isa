import sys
import unittest 

sys.path.append('./code')

from models import GSM
from tools import logmeanexp
from numpy import zeros, all, abs, array, square, log, pi, sum, mean, inf, exp
from numpy import histogram, max
from numpy.random import randn, rand
from scipy import integrate

class Tests(unittest.TestCase):
	def test_energy_gradient(self):
		"""
		Tests whether the energy gradient is similar to a numerical gradient.
		"""

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
		"""
		Tests whether 1-dimensional GSMs are normalized. Tests the log-likelihood
		of several instantiations of the GSM.
		"""

		# check whether the log-likelihood of 1D GSMs is normalized
		for num_scales in [1, 2, 3, 4, 5]:
			model = GSM(1, num_scales=num_scales)

			# implied probability density of model
			pdf = lambda x: exp(model.loglikelihood(array(x).reshape(1, -1)))

			# compute normalization constant and upper bound on error
			partf, err = integrate.quad(pdf, -inf, inf)

			self.assertTrue(partf - err <= 1.)
			self.assertTrue(partf + err >= 1.)

		# test the log-likelihood of a couple of GSMs
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



	def test_train(self):
		"""
		Tests whether training can recover parameters.
		"""

		for dim in [1, 2, 3]:
			gsm1 = GSM(dim, 2)
			gsm1.scales = array([0.5, 4.])

			data = gsm1.sample(20000)

			gsm2 = GSM(dim, 2)
			gsm2.train(data, max_iter=100)

			self.assertTrue(any(abs(gsm1.scales[0] - gsm2.scales) < 1E-1))
			self.assertTrue(any(abs(gsm1.scales[1] - gsm2.scales) < 1E-1))



	def test_sample(self):
		"""
		Compares model density with histogram obtained from samples.
		"""

		model = GSM(1, 3)
		model.scales = array([1., 3., 8.])

		data = model.sample(50000)

		try:
			hist, x = histogram(data, 100, density=True)
		except:
			# use deprecated method with older versions of Python
			hist, x = histogram(data, 100, normed=True)
		x = (x[1:] + x[:-1]) / 2.

		pdf = exp(model.loglikelihood(x.reshape(1, -1)))

		self.assertTrue(all(abs(pdf - hist) < 1E-1))



if __name__ == '__main__':
	unittest.main()
