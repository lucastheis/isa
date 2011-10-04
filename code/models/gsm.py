"""
An implementation of a simple isotropic GSM.
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@theis.io>'
__docformat__ = 'epytext'

from distribution import Distribution
from numpy.random import rand, randint, randn
from numpy import *
from scipy.stats import gamma
from utils import logmeanexp, logsumexp

class GSM(Distribution):
	def __init__(self, dim=1, num_scales=8):
		self.dim = dim
		self.num_scales = num_scales
		self.scales = 0.75 + rand(num_scales) / 2.
		self.scales /= mean(self.scales)



	def initialize(self, method='cauchy'):
		if method.lower() == 'student':
			# sample scales from respective Gamma distribution
			self.scales = 1. / sqrt(gamma.rvs(1, 0, 1, size=self.num_scales))

		elif method.lower() == 'cauchy':
			# sample scales from respective Gamma distribution
			self.scales = 1. / sqrt(gamma.rvs(0.5, 0, 2, size=self.num_scales))

		else:
			raise ValueError('Unknown initialization method \'{0}\'.'.format(method))

		# normalize scales
		self.scales /= mean(self.scales)



	def train(self, data):
		pass



	def sample(self, num_samples=1):
		# sample scales
		scales = self.scales[randint(self.num_scales, size=num_samples)]

		# sample data points
		return randn(self.dim, num_samples) * scales



	def sample_posterior(self, data):
		scales = self.scales.reshape(self.num_scales, 1)

		# calculate cumulative posterior over scales
		norms = sum(square(data), 0).reshape(1, -1)
		cmf = cumsum(exp(-0.5 * norms / square(scales) - self.dim * log(scales)), 0)
		cmf /= cmf[-1]

		# container for scale indices
		indices = zeros(data.shape[1], 'int32')

		# sample posterior
		uni = rand(data.shape[1])
		for j in range(self.num_scales - 1):
			indices[uni > cmf[j]] = j + 1

		return self.scales[indices]



	def loglikelihood(self, data):
		return self.energy(data) - self.dim / 2. * log(2. * pi)



	def energy(self, data):
		scales = self.scales.reshape(self.num_scales, 1)

		# compute unnormalized log-likelihoods
		norms = sum(square(data), 0).reshape(1, -1)
		uloglik = -0.5 * norms / square(scales) - self.dim * log(scales)

		# average over scales
		return -logmeanexp(uloglik, 0)



	def energy_gradient(self, data):
		scales = self.scales.reshape(self.num_scales, 1)

		# compute posterior over scales
		norms = sum(square(data), 0).reshape(1, -1)
		post = -0.5 * norms / square(scales) - self.dim * log(scales)
		post = exp(post - logsumexp(post, 0))

		# compute energy gradient
		return multiply(dot(1. / square(scales).T, post), data)
