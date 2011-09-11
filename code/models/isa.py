"""
An implementation of overcomplete ISA, a generalization of ICA.
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@theis.io>'
__docformat__ = 'epytext'

from distribution import Distribution
from numpy import *
from numpy.random import randint, randn, rand
from numpy.linalg import svd, pinv, inv
from scipy.linalg import solve
from tools import gaborf, mapp
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

		# linear features
		self.A = randn(self.num_visibles, self.num_hiddens)

		# subspace densities
		self.subspaces = [
			GSM(ssize) for i in range(num_hiddens / ssize)]



	def initialize(self, data=None, method='data'):
		"""
		Initializes linear features with more sensible values.

		@type  data: array_like
		@param data: data points stored in columns

		@type  method: string
		@param method: type of initialization ('data', 'gabor' or 'random')
		"""

		if method.lower() == 'data':
			# initialize features with data points
			if data is not None:
				self.A = data[:, randint(data.shape[1], size=self.num_hiddens)]

		elif method.lower() == 'gabor':
			# initialize features with Gabor filters
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



	def sample_scales(self, data):
		scales, k = [], 0

		for model in self.subspaces:
			s = model.sample_posterior(data[k:k + model.dim])

			# add scales for each dimension
			scales.extend(s for _ in range(model.dim))

			k += model.dim

		return vstack(scales)



	def sample_posterior(self, data, method='gibbs'):
		if self.num_hiddens == self.num_visibles:
			return dot(inv(self.A), data)

		if method.lower() == 'gibbs':
			return self.sample_posterior_gibbs(data)

		elif method.lower() in ['hmc', 'hamilton']:
			return self.sample_posterior_hmc(data)

		else:
			raise ValueError('Unknown sampling method \'{0}\'.'.format(method))



	def sample_nullspace(self, data, method='gibbs'):
		# nullspace basis
		B = svd(self.A)[2][self.num_visibles:, :]

		return dot(B, self.sample_posterior(data, method=method))




	def sample_posterior_gibbs(self, data, num_steps=10):
		# filter matrix and filter responses
		W = pinv(self.A)
		Wx = dot(W, data)

		# nullspace projection matrix
		Q = eye(self.num_hiddens) - dot(W, self.A)

		# initial hidden state
		y = asshmarray(Wx + dot(Q, self.sample_prior(data.shape[1])))

		for step in range(num_steps):
			if Distribution.VERBOSITY > 1:
				print step, mean(self.energy(y))

			# sample scales
			s = self.sample_scales(y)

			# sample from prior conditioned on scales
			y_ = multiply(randn(self.num_hiddens, data.shape[1]), s)
			x_ = data - dot(self.A, y_)

			# variances and partial covariances
			v = square(s).reshape(-1, 1, data.shape[1])
			C = multiply(v, self.A.T.reshape(self.num_hiddens, -1, 1)).transpose([2, 0, 1])

			# sample hidden state
			def parfor(i):
				y[:, i] = dot(C[i], solve(dot(self.A, C[i]), x_[:, i], sym_pos=True))
			mapp(parfor, range(data.shape[1]))
			y = asshmarray(Wx + dot(Q, y + y_))

		return asarray(y)



	def sample_posterior_hmc(self, X, num_steps=500):
		# filter matrix
		W = pinv(self.A)

		# nullspace basis
		B = svd(self.A)[2][self.num_visibles:, :]

		# nullspace projection matrix
		Q = dot(B.T, B)

		# sampling hyperparameters
		lf_num_steps = 5
		lf_step_sizes = 0.02 + zeros([1, X.shape[1]])
		mh_acc_rate = 0.95

		# initialization of hidden variables
		Y = dot(W, X)
		Z = dot(B, self.sample_prior(X.shape[1]))

		# perform hybrid Monte Carlo sampling
		for step in range(num_steps):
			if Distribution.VERBOSITY > 1:
				print step, mean(self.energy(Y + dot(B.T, Z)))

			# sample momentum
			P = randn(*Z.shape)

			# store Hamiltonian
			Zold = copy(Z)
			Hold = self.energy(Y + dot(B.T, Z)) + sum(square(P), 0) / 2.

			# leapfrog steps
			P -= lf_step_sizes / 2. * dot(B, self.energy_gradient(Y + dot(B.T, Z)))
			for _ in range(lf_num_steps):
				Z += lf_step_sizes * P
				tmp = dot(B, self.energy_gradient(Y + dot(B.T, Z)))
				P -= lf_step_sizes * tmp
			P += lf_step_sizes / 2. * tmp

			# new Hamiltonian
			Hnew = self.energy(Y + dot(B.T, Z)) + sum(square(P), 0) / 2.

			# Metropolis accept/reject step
			reject = rand(1, Z.shape[1]) > exp(Hold - Hnew)
			Z[:, reject] = Zold[:, reject]

			# adjust step sizes so that acceptance rate stays constant
			lf_step_sizes[reject] *= 0.8
			lf_step_sizes[-reject] *= 1. / power(0.8, (1. - mh_acc_rate) / mh_acc_rate)


		return Y + dot(B.T, Z)



	def energy_gradient(self, data):
		gradient, k = [], 0

		for model in self.subspaces:
			gradient.append(
				model.energy_gradient(data[k:k + model.dim]))
			k += model.dim

		return vstack(gradient)



	def energy(self, data):
		energy, k = zeros([1, data.shape[1]]), 0

		for model in self.subspaces:
			energy += model.energy(data[k:k + model.dim])
			k += model.dim

		return energy
