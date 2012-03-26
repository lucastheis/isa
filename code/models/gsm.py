"""
A lightweight implementation of an isotropic GSM.
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@theis.io>'
__docformat__ = 'epytext'

from distribution import Distribution
from numpy.random import rand, randint, randn
from numpy import *
from numpy import min, max
from scipy.stats import gamma, rayleigh
from tools import logmeanexp, logsumexp

class GSM(Distribution):
	def __init__(self, dim=1, num_scales=10):
		self.dim = dim
		self.num_scales = num_scales

		# standard deviations
		self.scales = 0.75 + rand(num_scales) / 2.
		self.scales /= mean(self.scales)

		# regularization parameters (inverse Gamma prior)
		self.alpha = 2. # shape
		self.beta = 1. # scale
		self.gamma = 0. # strength



	def initialize(self, method='cauchy'):
		if method.lower() == 'student':
			self.scales = 1. / sqrt(gamma.rvs(1, 0, 1, size=self.num_scales))

		elif method.lower() == 'cauchy':
			self.scales = 1. / sqrt(gamma.rvs(0.5, 0, 2, size=self.num_scales))

		elif method.lower() == 'laplace':
			self.scales = rayleigh.rvs(size=self.num_scales)

		else:
			raise ValueError('Unknown initialization method \'{0}\'.'.format(method))

		self.normalize()



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

		value = -mean(self.loglikelihood(data)) \
			+ self.gamma * (self.alpha + 1) * sum(log(self.scales)) \
			+ self.gamma / 2. * sum(self.beta / square(self.scales))
		
		if Distribution.VERBOSITY > 2:
			print 0, value

		# compute squared norms of data points
		sqnorms = sum(square(data), 0).reshape(1, -1)

		for i in range(max_iter):
			scales = self.scales.reshape(-1, 1)

			# calculate posterior over scales (E)
			post = -0.5 * sqnorms / square(scales) - self.dim * log(scales)
			post = exp(post - logsumexp(post, 0))

			try:
				# adjust parameters (M)
				self.scales = sqrt((mean(post * sqnorms, 1) + self.gamma * self.beta) / \
					(self.dim * mean(post, 1) + self.gamma * (self.alpha + 1)))

			except FloatingPointError:
				indices, = where(sum(post, 1) == 0.)

				if Distribution.VERBOSITY > 0:
					print 'Degenerated scales {0}.'.format(self.scales[indices])

				# reset problematic scales
				self.scales[indices] = 0.75 + rand(len(indices)) / 2.  
				value = self.evaluate(data)

			# check for convergence
			value_ = -mean(self.loglikelihood(data)) \
				+ self.gamma * (self.alpha + 1.) * sum(log(self.scales)) \
				+ self.gamma / 2. * sum(self.beta / square(self.scales))
			if value - value_ < tol:
				break
			value = value_

			if Distribution.VERBOSITY > 2:
				print i + 1, value



	def normalize(self):
		"""
		Normalizes the scales so that the standard deviation of the GSM becomes 1.
		"""

		self.scales /= sqrt(mean(square(self.scales)))



	def std(self):
		"""
		Returns the variance of this Gaussian scale mixture.
		"""

		return sqrt(mean(square(self.scales)))



	def sample(self, num_samples=1):
		"""
		Generate data samples.
		"""

		# sample scales
		scales = self.scales[randint(self.num_scales, size=num_samples)]

		# sample data points
		return randn(self.dim, num_samples) * scales



	def sample_posterior(self, data):
		"""
		Draw samples from posterior over scales.
		"""

		scales = self.scales.reshape(-1, 1)

		# calculate cumulative posterior over scales
		sqnorms = sum(square(data), 0).reshape(1, -1)
		cmf = cumsum(exp(-0.5 * sqnorms / square(scales) - self.dim * log(scales)), 0)
		cmf /= cmf[-1]

		# container for scale indices
		indices = zeros(data.shape[1], 'int32')

		# sample posterior
		uni = rand(data.shape[1])
		for j in range(self.num_scales - 1):
			indices[uni > cmf[j]] = j + 1

		return self.scales[indices].reshape(1, -1)



	def posterior(self, data):
		"""
		Calculate posterior over scales.

		@type  data: array_like
		@param data: data points stored in columns

		@type: ndarray
		@return: posterior over scales
		"""

		scales = self.scales.reshape(-1, 1)
		sqnorms = sum(square(data), 0).reshape(1, -1)
		post = exp(-0.5 * sqnorms / square(scales) - self.dim * log(scales))
		post /= sum(post, 0)

		return post



	def loglikelihood(self, data):
		return -self.energy(data) - self.dim / 2. * log(2. * pi)



	def energy(self, data):
		scales = self.scales.reshape(self.num_scales, 1)

		# compute unnormalized log-likelihoods
		sqnorms = sum(square(data), 0).reshape(1, -1)
		uloglik = -0.5 * sqnorms / square(scales) - self.dim * log(scales)

		# average over scales
		return -logmeanexp(uloglik, 0).reshape(1, -1)



	def energy_gradient(self, data):
		scales = self.scales.reshape(self.num_scales, 1)

		# compute posterior over scales
		sqnorms = sum(square(data), 0).reshape(1, -1)

		# slow, but stable
		post = -0.5 * sqnorms / square(scales) - self.dim * log(scales)
		post = exp(post - logsumexp(post, 0))

		# compute energy gradient
		return multiply(dot(1. / square(scales).T, post), data)
