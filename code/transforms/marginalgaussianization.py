__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@theis.io>'
__docformat__ = 'epytext'

from transform import Transform
from univariategaussianization import UnivariateGaussianization
from numpy.linalg import inv, slogdet
from numpy import vstack, dot, zeros

class MarginalGaussianization(Transform):
	def __init__(self, ica):
		"""
		@type  ica: L{ICA}
		@param ica: ICA model used for Gaussianization
		"""

		self.ica = ica



	def apply(self, data):
		"""
		@type  data: array_like
		@param data: data points stored in columns
		"""

		# linearly transform data
		data = dot(inv(self.ica.A), data)

		length = len(str(len(self.ica.marginals)))

		if Transform.VERBOSITY > 0:
			print ('{0:>' + str(length) + '}/{1}').format(0, len(self.ica.marginals)),

		for i, mog in enumerate(self.ica.marginals):
			data[i] = UnivariateGaussianization(mog).apply(data[[i]])

			if Transform.VERBOSITY > 0:
				print (('\b' * (length * 2 + 2)) + '{0:>' + str(length) + '}/{1}').format(i + 1, len(self.ica.marginals)),
		if Transform.VERBOSITY > 0:
			print

		return data



	def inverse(self, data):
		"""
		Apply inverse Gaussianization.
		"""

		data_irg = []

		length = len(str(len(self.ica.marginals)))

		if Transform.VERBOSITY > 0:
			print ('{0:>' + str(length) + '}/{1}').format(0, len(self.ica.marginals)),

		for i, mog in enumerate(self.ica.marginals):
			data[i] = UnivariateGaussianization(mog).inverse(data[[i]])

			if Transform.VERBOSITY > 0:
				print (('\b' * (length * 2 + 2)) + '{0:>' + str(length) + '}/{1}').format(i + 1, len(self.ica.marginals)),
		if Transform.VERBOSITY > 0:
			print

		# linearly transform data
		return dot(self.ica.A, data)



	def logjacobian(self, data):
		"""
		Returns the log-determinant of the Jabian matrix evaluated at the given
		data points.

		@type  data: array_like
		@param data: data points stored in columns

		@rtype: ndarray
		@return: the logarithm of the Jacobian determinants
		"""

		# completed filter matrix
		W = inv(self.ica.A)

		# determinant of linear transformation
		logjacobian = zeros([1, data.shape[1]]) + slogdet(W)[1]

		# linearly transform data
		data = dot(W, data)

		length = len(str(len(self.ica.marginals)))

		if Transform.VERBOSITY > 0:
			print ('{0:>' + str(length) + '}/{1}').format(0, len(self.ica.marginals)),

		for i, mog in enumerate(self.ica.marginals):
			logjacobian += UnivariateGaussianization(mog).logjacobian(data[[i]])

			if Transform.VERBOSITY > 0:
				print (('\b' * (length * 2 + 2)) + '{0:>' + str(length) + '}/{1}').format(i + 1, len(self.ica.marginals)),
		if Transform.VERBOSITY > 0:
			print

		return logjacobian
