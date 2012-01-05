"""
A lightweight implementation of a univariate mixture of Gaussians.
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@theis.io>'
__docformat__ = 'epytext'

from distribution import Distribution
from numpy import ones, square, sum, multiply, log, exp, mean, std, where, sqrt, pi, round
from numpy import cumsum, zeros
from numpy.random import randn, rand, multinomial, permutation
from scipy.stats import gamma, rayleigh
from tools import logsumexp

class MoGaussian(Distribution):
	def __init__(self, num_components=8):
		self.num_components = num_components

		# regularization of prior weights
		self.alpha = None

		# prior weights, means and standard deviations
		self.priors = ones(num_components) / num_components
		self.means = randn(num_components) / 100.
		self.scales = 0.75 + rand(num_components) / 2.



	def initialize(self, method='laplace'):
		"""
		Randomly initializes parameters.
		"""

		if method.lower() == 'student':
			self.scales = 1. / sqrt(gamma.rvs(1, 0, 1, size=self.num_components))
			self.means *= 0.

		elif method.lower() == 'cauchy':
			self.scales = 1. / sqrt(gamma.rvs(0.5, 0, 2, size=self.num_components))
			self.means *= 0.

		elif method.lower() == 'laplace':
			self.scales = rayleigh.rvs(size=self.num_components)
			self.means *= 0.

		else:
			raise ValueError('Unknown initialization method \'{0}\'.'.format(method))



	def train(self, data, max_iter=10, tol=1e-5):
		"""
		Fits the parameters to the given data.

		@type  data: array_like
		@param data: data stored in columns

		@type  max_iter: integer
		@param max_iter: the maximum number of EM iterations

		@type  tol: float
		@param tol: stop if performance improves less than this threshold
		"""

		value = self.evaluate(data)

		if Distribution.VERBOSITY > 2:
			print 0, value

		# make sure data has the right shape
		data = data.reshape(1, -1)

		for i in range(max_iter):
			# reshape parameters
			priors = self.priors.reshape(-1, 1)
			means = self.means.reshape(-1, 1)
			scales = self.scales.reshape(-1, 1)

			# calculate posterior (E)
			post = log(priors) - 0.5 * square(data - means) / square(scales) - log(scales)
			post = exp(post - logsumexp(post, 0))

			try:
				weights = post / sum(post, 1).reshape(-1, 1)

			except FloatingPointError:
				if Distribution.VERBOSITY > 0:
					print 'Mixture with zero posterior probability detected.'

				indices, = where(sum(post, 1) == 0.)

				# reset problematic components
				self.means[indices] = mean(data) + randn(len(indices)) / 100.
				self.scales[indices] = std(data) * (0.75 + rand(len(indices)) / 2.)
				value = self.evaluate(data)

				continue

			# update parameters (M)
			self.priors = sum(post, 1) / sum(post)
			self.means = sum(multiply(data, weights), 1)
			self.scales = sqrt(sum(multiply(square(data - self.means.reshape(-1, 1)), weights), 1))

			# regularize priors
			if self.alpha is not None:
				self.priors = self.priors + self.alpha
				self.priors = self.priors / sum(self.priors)

			# check for convergence
			value_ = self.evaluate(data)
			if value - value_ < tol:
				break
			value = value_

			if Distribution.VERBOSITY > 2:
				print i + 1, value



	def sample(self, num_samples=1):
		samples = randn(1, num_samples)
		samples_ = samples

		num_samples = multinomial(num_samples, self.priors)

		for i in range(self.num_components):
			samples_[:, :num_samples[i]] *= self.scales[i]
			samples_[:, :num_samples[i]] += self.means[i]
			samples_ = samples_[:, num_samples[i]:]

		samples = samples[:, permutation(samples.shape[1])]

		return samples



	def loglikelihood(self, data):
		# make sure data has right shape
		data = data.reshape(1, -1)

		return -self.energy(data) - 0.5 * log(2. * pi)



	def energy(self, data):
		# make sure data has right shape
		data = data.reshape(1, -1)

		# reshape parameters
		priors = self.priors.reshape(-1, 1)
		means = self.means.reshape(-1, 1)
		scales = self.scales.reshape(-1, 1)

		# joint density of indices and data
		joint = log(priors) - 0.5 * square(data - means) / square(scales) \
			- log(scales)

		return -logsumexp(joint, 0).reshape(1, -1)



	def energy_gradient(self, data):
		# make sure data has right shape
		data = data.reshape(1, -1)

		# reshape parameters
		priors = self.priors.reshape(-1, 1)
		means = self.means.reshape(-1, 1)
		scales = self.scales.reshape(-1, 1)

		data_centered = data - means

		# calculate posterior
		post = log(priors) - 0.5 * square(data_centered) / square(scales) - log(scales)
		post = exp(post - logsumexp(post, 0))

		return sum(multiply(data_centered / square(scales), post), 0).reshape(1, -1)



	def posterior(self, data):
		"""
		Calculate posterior over mixture components for each given data point.

		@type  data: array_like
		@param data: data points

		@type: ndarray
		@return: posterior over mixture components
		"""

		# make sure data has right shape
		data = data.reshape(1, -1)

		# reshape parameters
		priors = self.priors.reshape(-1, 1)
		means = self.means.reshape(-1, 1)
		scales = self.scales.reshape(-1, 1)

		data_centered = data - means

		# calculate posterior
		post = log(priors) - 0.5 * square(data_centered) / square(scales) - log(scales)
		post = exp(post - logsumexp(post, 0))

		return post



	def sample_posterior(self, data):
		"""
		Samples means and standard deviations from the posterior for the given data points.

		@type  data: array_like
		@param data: data points stored in columns

		@rtype: tuple
		@return: means and standard deviations for each data point
		"""

		# make sure data has right shape
		data = data.reshape(1, -1)

		cmf = cumsum(self.posterior(data), 0)

		# component indices
		indices = zeros(data.shape[1], 'int32')

		# sample posterior
		uni = rand(data.shape[1])
		for j in range(self.num_components - 1):
			indices[uni > cmf[j]] = j + 1

		return self.means[indices].reshape(1, -1), \
			self.scales[indices].reshape(1, -1)
