__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@theis.io>'
__docformat__ = 'epytext'

from transform import Transform
from scipy.stats import chi
from scipy.special import gamma
from scipy.optimize import bisect
from tools import gammaincinv, logsumexp
from numpy import sqrt, sum, square, multiply, zeros_like, zeros, log

class RadialGaussianization(Transform):
	def __init__(self, gsm):
		"""
		@type  gsm: L{GSM}
		@param gsm: Gaussian scale mixture used for Gaussianization
		"""

		self.gsm = gsm

	

	def apply(self, data):
		"""
		Radially Gaussianizes the given data.

		@type  data: array_like
		@param data: data points stored in columns
		"""

		def rcdf(norm):
			"""
			Radial cumulative distribution function (CDF).
			"""

			# allocate memory
			result = zeros_like(norm)

			for j in range(self.gsm.num_scales):
				result += grcdf(norm / self.gsm.scales[j], self.gsm.dim)
			result /= self.gsm.num_scales
			result[result > 1.] = 1.

			return result

		# radially Gaussianize data
		norm = sqrt(sum(square(data), 0))
		return multiply(igrcdf(rcdf(norm), self.gsm.dim) / norm, data)




	def inverse(self, data, maxiter=100):
		"""
		Applies the inverse transformation to the given set of data points.

		@type  data: array_like
		@param data: data points stored in columns
		"""

		def rcdf(norm):
			"""
			Radial cumulative distribution function for real values.

			@type  norm: float
			@param norm: one-dimensional, positive input
			"""
			return sum(grcdf(norm / self.gsm.scales, self.gsm.dim)) / self.gsm.num_scales

		# compute norm
		norm = sqrt(sum(square(data), 0))

		# normalize data
		data = data / norm

		# apply Gaussian radial CDF
		norm = grcdf(norm, self.gsm.dim)

		# apply inverse radial CDF
		norm_max = 1.
		for t in range(len(norm)):
			# make sure root lies between zero and norm_max
			while rcdf(norm_max) < norm[t]:
				norm_max += 1.
			# find root numerically
			norm[t] = bisect(
			    f=lambda x: rcdf(x) - norm[t],
			    a=0.,
			    b=norm_max,
			    maxiter=maxiter,
			    disp=False)

		# inverse radial Gaussianization
		data = multiply(norm, data)

		return data



	def logjacobian(self, data):
		"""
		Returns the log-determinant of the Jacobian of radial Gaussianization
		evaluated at the given data points.

		@type  data: array_like
		@param data: data points stored in columns

		@rtype: ndarray
		@return: the log-Jacobian determinants
		"""

		def rcdf(norm):
			"""
			Radial cumulative distribution function (CDF).
			"""

			# allocate memory
			result = zeros_like(norm)

			for j in range(self.gsm.num_scales):
				result += grcdf(norm / self.gsm.scales[j], self.gsm.dim)
			result /= self.gsm.num_scales
			result[result > 1.] = 1.

			return result


		def logdrcdf(norm):
			"""
			Logarithm of the derivative of the radial CDF.
			"""

			# allocate memory
			result = zeros([self.gsm.num_scales, len(norm)])

			for j in range(self.gsm.num_scales):
				result[j, :] = logdgrcdf(norm / self.gsm.scales[j], self.gsm.dim) - log(self.gsm.scales[j])
			result -= log(self.gsm.num_scales)

			return logsumexp(result, 0)

		# data norm
		norm = sqrt(sum(square(data), 0))

		# radial gaussianization function applied to the norm
		tmp1 = igrcdf(rcdf(norm), self.gsm.dim)

		# log of derivative of radial gaussianization function
		logtmp2 = logdrcdf(norm) - logdgrcdf(tmp1, self.gsm.dim)

		# return log of Jacobian determinant
		return (self.gsm.dim - 1) * log(tmp1 / norm) + logtmp2




def grcdf(norm, dim):
	"""
	Gaussian radial CDF.
	
	@type  norm: array_like
	@param norm: norms of the data points

	@type  dim: integer
	@param dim: dimensionality of the Gaussian
	"""

	return chi.cdf(norm, dim)



def igrcdf(norm, dim):
	"""
	Inverse Gaussian radial CDF.

	@type  norm: array_like
	@param norm: norms of the data points

	@type  dim: integer
	@param dim: dimensionality of the Gaussian
	"""

	return sqrt(2.) * sqrt(gammaincinv(dim / 2., norm))



def igrcdf(norm, dim):
	"""
	Inverse Gaussian radial CDF.
	
	@type  norm: array_like
	@param norm: norms of the data points

	@type  dim: integer
	@param dim: dimensionality of the Gaussian
	"""

	return sqrt(2.) * sqrt(gammaincinv(dim / 2., norm))




def logdgrcdf(norm, dim):
	"""
	Logarithm of the derivative of the Gaussian radial CDF.
	
	@type  norm: array_like
	@param norm: norms of the data points

	@type  dim: integer
	@param dim: dimensionality of the Gaussian
	"""

	tmp = square(norm) / 2.
	return (dim / 2. - 1.) * log(tmp) - tmp - log(gamma(dim / 2)) + log(norm)
