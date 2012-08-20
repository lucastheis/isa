__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@theis.io>'
__docformat__ = 'epytext'

from transform import Transform
from radialgaussianization import RadialGaussianization
from numpy.linalg import inv, slogdet
from numpy import vstack, dot, zeros
from collections import Callable

class SubspaceGaussianization(Transform):
	def __init__(self, isa):
		"""
		@type  gsm: L{ISA}
		@param gsm: ISA model used for Gaussianization
		"""

		self.isa = isa



	def apply(self, data):
		"""
		Subspace Gaussianize data by first applying a linear transformation and then
		radially Gaussianizing each subspace. If C{isa} is overcomplete, C{data} has
		to be completed by the nullspace representation, that is, C{data} should
		have the dimensionality of the hidden states.

		@type  data: array_like
		@param data: data points stored in columns
		"""

		# complete basis
		A = vstack([self.isa.A, self.isa.nullspace_basis()])

		# linearly transform data
		data = dot(inv(A), data)

		data_rg = []

		if isinstance(self.isa.subspaces, Callable):
			subspaces = self.isa.subspaces()
		else:
			subspaces = self.isa.subspaces

		length = len(str(len(subspaces)))

		if Transform.VERBOSITY > 0:
			print ('{0:>' + str(length) + '}/{1}').format(0, len(subspaces)),

		# TODO: parallelize
		for i, gsm in enumerate(subspaces):
			# radially Gaussianize subspace
			data_rg.append(
				RadialGaussianization(gsm).apply(data[:gsm.dim]))
			data = data[gsm.dim:]

			if Transform.VERBOSITY > 0:
				print (('\b' * (length * 2 + 2)) + '{0:>' + str(length) + '}/{1}').format(i + 1, len(subspaces)),
		if Transform.VERBOSITY > 0:
			print

		return vstack(data_rg)



	def inverse(self, data):
		"""
		Apply inverse subspace Gaussianization.
		"""

		data_irg = []

		if isinstance(self.isa.subspaces, Callable):
			subspaces = self.isa.subspaces()
		else:
			subspaces = self.isa.subspaces

		length = len(str(len(subspaces)))

		if Transform.VERBOSITY > 0:
			print ('{0:>' + str(length) + '}/{1}').format(0, len(subspaces)),

		# TODO: parallelize
		for i, gsm in enumerate(subspaces):
			# inverse radially Gaussianize subspace
			data_irg.append(
				RadialGaussianization(gsm).inverse(data[:gsm.dim]))
			data = data[gsm.dim:]

			if Transform.VERBOSITY > 0:
				print (('\b' * (length * 2 + 2)) + '{0:>' + str(length) + '}/{1}').format(i + 1, len(subspaces)),
		if Transform.VERBOSITY > 0:
			print

		data = vstack(data_irg)

		# completed filter matrix
#		A = vstack([self.isa.A, self.isa.nullspace_basis()])

		# linearly transform data
		return dot(self.isa.A, data)



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
		A = vstack([self.isa.A, self.isa.nullspace_basis()])
		W = inv(A)

		# determinant of linear transformation
		logjacobian = zeros([1, data.shape[1]]) + slogdet(W)[1]

		# linearly transform data
		data = dot(W, data)

		if isinstance(self.isa.subspaces, Callable):
			subspaces = self.isa.subspaces()
		else:
			subspaces = self.isa.subspaces

		length = len(str(len(subspaces)))

		if Transform.VERBOSITY > 0:
			print ('{0:>' + str(length) + '}/{1}').format(0, len(subspaces)),

		# TODO: parallelize
		for i, gsm in enumerate(subspaces):
			logjacobian += RadialGaussianization(gsm).logjacobian(data[:gsm.dim])
			data = data[gsm.dim:]

			if Transform.VERBOSITY > 0:
				print (('\b' * (length * 2 + 2)) + '{0:>' + str(length) + '}/{1}').format(i + 1, len(subspaces)),
		if Transform.VERBOSITY > 0:
			print

		return logjacobian
