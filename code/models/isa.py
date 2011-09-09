"""
An implementation of overcomplete ISA, a generalization of ICA.
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@theis.io>'
__docformat__ = 'epytext'

from distribution import Distribution
from numpy import *
from numpy.random import randint, randn, rand
from numpy.linalg import svd, pinv, inv, cholesky
from scipy.linalg import solve
from tools import gaborf
from tools.parallel import map
from tools.shmarray import asshmarray
from gsm import GSM

class ISA(Distribution):
	"""
	An implementation of overcomplete ISA.
	"""

	def __init__(self, num_visibles, num_hiddens=None, ssize=1):
		"""
		"""

		if mod(num_hiddens, ssize):
			raise ValueError('num_hiddens must be a multiple of ssize.')

		self.num_visibles = num_visibles
		self.num_hiddens = num_hiddens

		if not num_hiddens:
			self.num_hiddens = num_visibles

		# linear filters
		self.A = randn(self.num_visibles, self.num_hiddens)

		# subspace densities
		self.subspaces = [
			GSM(ssize) for i in range(num_hiddens / ssize)]



	def initialize(self, data=None, method='data'):
		"""
		Initializes linear filters with more sensible values.

		@type  data: array_like
		@param data: data points stored in columns

		@type  method: string
		@param method: type of initialization ('data', 'gabor' or 'random')
		"""

		if method.lower() == 'data':
			# initialize filters with data points
			if data is not None:
				self.A = data[:, randint(data.shape[1], size=self.num_hiddens)]

		elif method.lower() == 'gabor':
			# initialize filters with Gabor filters
			if self.subspaces[0].dim > 1 and not mod(self.num_hiddens, 2):
				for i in range(self.num_hiddens / 2):
					G = gaborf(self.num_visibles)
					self.A[:, 2 * i] = real(G)
					self.A[:, 2 * i + 1] = imag(G)
			else:
				for i in range(len(self.subspaces)):
					self.A[:, i] = gaborf(self.num_visibles, complex=False)

			if data is not None:
				dot(pinv(self.A), data)

		elif method.lower() == 'random':
			# initialize with Gaussian white noise
			self.A = randn(num_visibles, num_hiddens)

		else:
			raise ValueError('Unknown initialization method \'{0}\'.'.format(method))

		# initialize subspace distributions
		for model in self.subspaces:
			model.initialize()



	def train(self, data, method=''):
		pass



	def sample(self, num_samples=1):
		"""
		Draw samples from the model.

		@type  num_samples: integer
		@param num_samples: number of samples to draw

		@rtype: array
		@return: array with num_samples columns
		"""

		return dot(self.A, self.sample_prior(num_samples))



	def sample_prior(self, num_samples=1):
		return vstack(m.sample(num_samples) for m in self.subspaces)



	def sample_scales(self, y):
		s = []
		k = 0
		for model in self.subspaces:
			scales = model.sample_posterior(y[k:k + model.dim])
			for j in range(model.dim):
				s.append(scales)
			k += model.dim
		s = vstack(s)

		return s



	def sample_posterior(self, data, num_steps=5):
		if self.num_hiddens == self.num_visibles:
			return dot(inv(self.A), data)

		# create orthogonal basis for nullspace
		A = self.A
		B = svd(A)[2][self.num_visibles:, :]

		VB = dot(B.T, B)

		Wx = dot(pinv(self.A), data)

		y = asshmarray(Wx + dot(VB, self.sample_prior(data.shape[1])))

		for _ in range(num_steps):
			# sample scales
			s = self.sample_scales(y)

			# sample from prior conditioned on scales
			y_ = multiply(randn(self.num_hiddens, data.shape[1]), s)
			x_ = data - dot(A, y_)

			C = multiply(
				square(s).reshape(-1, 1, data.shape[1]),
				A.T.reshape(self.num_hiddens, -1, 1)).transpose([2, 0, 1])

			def parfor(i):
				y[:, i] = dot(C[i], solve(dot(A, C[i]), x_[:, i], sym_pos=True))

			map(parfor, range(data.shape[1]))

			y = asshmarray(Wx + dot(VB, asarray(y) + y_))

		return y
