"""
Provides an interface for all transforms.
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@bethgelab.org>'
__docformat__ = 'epytext'

class Transform(object):
	def __init__(self):
		raise NotImplementedError(str(self.__class__) + ' is an abstract class.')



	def __call__(self, data):
		return self.apply(data)



	def apply(self, data):
		"""
		Applies the transformation to the given set of data points.

		@type  data: array_like
		@param data: data points stored in columns
		"""

		raise NotImplementedError('Abstract method \'apply\' not implemented in '
			+ str(self.__class__))



	def inverse(self, data):
		"""
		Applies the inverse transformation to the given set of data points.

		@type  data: array_like
		@param data: data points stored in columns
		"""

		raise NotImplementedError('Abstract method \'inverse\' not implemented in '
			+ str(self.__class__))



	def logjacobian(self, data):
		"""
		Returns the log-determinant of the Jacobian evaluated at the given data points.

		@type  data: array_like
		@param data: data points stored in columns

		@rtype: ndarray
		@return: the logarithm of the Jacobian determinants
		"""

		raise NotImplementedError('Abstract method \'logjacobian\' not implemented in '
			+ str(self.__class__))
