__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@theis.io>'
__docformat__ = 'epytext'

from transform import Transform
from numpy import *
from numpy.linalg import slogdet, inv
from scipy import indices

class LinearTransform(Transform):
	def __init__(self, *args, **kwargs):
		"""
		A linear transformation. If initialized with the DCT basis, the first
		feature corresponds to the DC component.

		@type  A: array_like
		@param A: linear transform matrix

		@type  basis: string
		@param basis: currently only 'DCT' is an available basis

		@type  dim: integer
		@param dim: dimensionality of basis
		"""

		if 'A' in kwargs:
			self.A = asarray(kwargs['A'])
			self.dim = self.A.shape[0]

		if len(args) > 0:
			if 'basis' in kwargs:
				raise ValueError('Did you forget to use the `dim` keyword?')
			self.A = asarray(args[0])
			self.dim = self.A.shape[0]

		elif 'basis' in kwargs:
			if 'dim' not in kwargs:
				raise ValueError('Please specify a dimensionality, `dim`.')

			self.dim = kwargs['dim']

			if kwargs['basis'].upper() == 'DCT':
				I, J = indices([kwargs['dim'], kwargs['dim']])

				A = []

				for p in range(kwargs['dim']):
					for q in range(kwargs['dim']):
						F = 2. * multiply(
							cos(pi * (2. * I + 1.) * p / (2. * kwargs['dim'])),
							cos(pi * (2. * J + 1.) * q / (2. * kwargs['dim']))) / kwargs['dim']

						if p == 0:
							F /= sqrt(2.)

						if q == 0:
							F /= sqrt(2.)

						A.append(F.reshape(1, -1))

				self.A = vstack(A)
		else:
			raise ValueError('Please specify a linear transform.')



	def apply(self, data):
		return dot(self.A, data)



	def inverse(self, data):
		return dot(inv(self.A), data)



	def logjacobian(self, data=None):
		if data is None:
			return slogdet(self.A)[1]
		return slogdet(self.A)[1] + zeros([1, data.shape[1]])
