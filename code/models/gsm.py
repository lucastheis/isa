"""
An implementation of a simple isotropic GSM.
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@theis.io>'
__docformat__ = 'epytext'

from distribution import Distribution
from numpy.random import rand, randint, randn
from numpy import *
from numpy import min, max
from scipy.stats import gamma
from utils import logmeanexp, logsumexp

class GSM(Distribution):
	def __init__(self, dim=1, num_scales=20):
		self.dim = dim
		self.num_scales = num_scales

		# standard deviations
		self.scales = 0.75 + rand(num_scales) / 2.
		self.scales /= mean(self.scales)



	def initialize(self, method='student'):
		if method.lower() == 'student':
			# sample scales using the Gamma distribution
			self.scales = 1. / sqrt(gamma.rvs(1, 0, 1, size=self.num_scales))

		elif method.lower() == 'cauchy':
			# sample scales using the Gamma distribution
			self.scales = 1. / sqrt(gamma.rvs(0.5, 0, 2, size=self.num_scales))

		else:
			raise ValueError('Unknown initialization method \'{0}\'.'.format(method))

		self.normalize()



	def train(self, data, max_iter=10):
		"""
		Estimates parameters using EM.
		"""
		
		if Distribution.VERBOSITY > 1:
			print 0, self.evaluate(data) / log(2.)

		for i in range(max_iter):
			scales = self.scales.reshape(-1, 1)

			# calculate posterior over scales (E)
			sqnorms = sum(square(data), 0).reshape(1, -1)
			post = exp(-0.5 * sqnorms / square(scales) - self.dim * log(scales))
			post /= sum(post, 0)

			# assign equal probability where all scales have near-zero probability
			post[isnan(post)] = 1. / self.num_scales

			# adjust parameters (M)
			self.scales = sqrt(sum(post * sqnorms, 1) / sum(post, 1) / self.dim)

			if Distribution.VERBOSITY > 1:
				print i + 1, self.evaluate(data) / log(2.)



	def normalize(self):
		"""
		Normalizes the scales so that the standard deviation of the GSM becomes 1.
		"""

		self.scales /= sqrt(mean(square(self.scales)))



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

		return self.scales[indices]



	def posterior(self, data):
		"""
		Calculate posterior over scales.
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
		return -logmeanexp(uloglik, 0)



	def energy_gradient(self, data):
		scales = self.scales.reshape(self.num_scales, 1)

		# compute posterior over scales
		sqnorms = sum(square(data), 0).reshape(1, -1)

		# faster but less stable
#		post = exp(-0.5 * sqnorms / square(scales) - self.dim * log(scales))
#		post /= sum(post, 0)

		# slower but more stable
		post = -0.5 * sqnorms / square(scales) - self.dim * log(scales)
		post = exp(post - logsumexp(post, 0))

		# compute energy gradient
		return multiply(dot(1. / square(scales).T, post), data)
