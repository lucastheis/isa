"""
An implementation of overcomplete ISA, a generalization of ICA.
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@theis.io>'
__docformat__ = 'epytext'

from distribution import Distribution
from numpy import *
from numpy import max
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



	def initialize(self, X=None, method='data'):
		"""
		Initializes linear features with more sensible values.

		@type  X: array_like
		@param X: data points stored in columns

		@type  method: string
		@param method: type of initialization ('data', 'gabor' or 'random')
		"""

		if method.lower() == 'data':
			# initialize features with data points
			if X is not None:
				self.A = X[:, randint(X.shape[1], size=self.num_hiddens)]

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
				dot(pinv(self.A), X)

		elif method.lower() == 'random':
			# initialize with Gaussian white noise
			self.A = randn(num_visibles, num_hiddens)

		else:
			raise ValueError('Unknown initialization method \'{0}\'.'.format(method))

		# initialize subspace distributions
		for model in self.subspaces:
			model.initialize()



	def train(self, X, method=''):
		pass



	def sample(self, num_samples=1):
		"""
		Draw samples from the model.

		@type  num_samples: integer
		@param num_samples: number of samples to draw

		@rtype: array
		@return: array with `num_samples` columns
		"""

		return dot(self.A, self.sample_prior(num_samples))



	def sample_prior(self, num_samples=1):
		"""
		Draw samples from the prior distribution over hidden units.

		@type  num_samples: integer
		@param num_samples: number of samples to draw

		@rtype: array
		@return: array with `num_hiddens` rows and `num_samples` columns
		"""

		return vstack(m.sample(num_samples) for m in self.subspaces)



	def sample_scales(self, Y):
		"""
		Samples scales for given states of the hidden units.

		@type  Y: array_like
		@param Y: states for the hidden units

		@rtype: array
		@return: array with `num_scales` rows and as many columns as `Y`
		"""

		scales = []

		for model in self.subspaces:
			# repeat sampled scales for all dimensions
			scales.extend(
				tile(model.sample_posterior(Y[:model.dim]), [model.dim, 1]))
			Y = Y[model.dim:]

		return vstack(scales)



	def sample_posterior(self, X, method=('gibbs', {})):
		"""
		Draw samples from the posterior distribution over hidden units.

		@type  X: array_like
		@param X: states for the visible units

		@type  method: tuple
		@param method: tuple consisting of method and keyword arguments

		@rtype: array
		@return: array with `num_hiddens` rows and as many columns as `X`
		"""

		if isinstance(method, str):
			method = (method, {})

		if len(method) < 2:
			method = (method[0], {})

		if self.num_hiddens == self.num_visibles:
			return dot(inv(self.A), X)

		if method[0].lower() == 'gibbs':
			return self.sample_posterior_gibbs(X, **method[1])

		elif method[0].lower() in ['hmc', 'hamilton']:
			return self.sample_posterior_hmc(X, **method[1])

		elif method[0].lower() == 'metropolis':
			return self.sample_posterior_metropolis(X, **method[1])

		else:
			raise ValueError('Unknown sampling method \'{0}\'.'.format(method))



	def sample_nullspace(self, X, method=('gibbs', {})):
		"""
		Draws a sample from the posterior over the linear model's null space.
		"""

		# nullspace basis
		B = svd(self.A)[2][self.num_visibles:, :]

		return dot(B, self.sample_posterior(X, method=method))




	def sample_posterior_gibbs(self, X, num_steps=10, Y=None):
		# filter matrix and filter responses
		W = pinv(self.A)
		Wx = dot(W, X)

		# nullspace projection matrix
		Q = eye(self.num_hiddens) - dot(W, self.A)

		# initial hidden state
		Y = asshmarray(Wx + dot(Q, Y)) if Y is not None else \
			asshmarray(Wx + dot(Q, self.sample_prior(X.shape[1])))

		for step in range(num_steps):
			if Distribution.VERBOSITY > 1:
				print step, mean(self.energy(Y))

			# update scales
			S = self.sample_scales(Y)

			# sample from prior conditioned on scales
			Y_ = multiply(randn(self.num_hiddens, X.shape[1]), S)
			X_ = X - dot(self.A, Y_)

			# variances and partial covariances
			v = square(S).reshape(-1, 1, X.shape[1])
			C = multiply(v, self.A.T.reshape(self.num_hiddens, -1, 1)).transpose([2, 0, 1])

			# update hidden state
			def parfor(i):
				Y[:, i] = dot(C[i], solve(dot(self.A, C[i]), X_[:, i], sym_pos=True))
			mapp(parfor, range(X.shape[1]))
			Y = asshmarray(Wx + dot(Q, Y + Y_))

		return asarray(Y)



	def sample_posterior_hmc(self, X, num_steps=100, Y=None, **kwargs):
		# hyperparameters
		acc_rate = kwargs.get('acc_rate', 0.9)
		adaptive = kwargs.get('adaptive', True)
		lf_num_steps = kwargs.get('lf_num_steps', 10)
		lf_step_sizes = kwargs.get('lf_step_size', 0.01) + zeros([1, X.shape[1]])
		lf_step_size_min = kwargs.get('lf_step_size_min', 0.001)
		lf_step_size_max = kwargs.get('lf_step_size_max', 0.2)
		lf_step_size_dec = kwargs.get('lf_step_size_dec', 0.95)
		lf_step_size_inc = kwargs.get('lf_step_size_inc', 
			1. / power(lf_step_size_dec, (1. - acc_rate) / acc_rate))

		# nullspace basis
		B = svd(self.A)[2][self.num_visibles:, :]

		# initialization of nullspace hidden variables
		Z = dot(B, Y) if Y is not None else \
			dot(B, self.sample_prior(X.shape[1]))
		Y = dot(pinv(self.A), X)

		for step in range(num_steps):
			# sample momentum
			P = randn(*Z.shape)

			# store Hamiltonian
			Zold = copy(Z)
			Hold = self.energy(Y + dot(B.T, Z)) + sum(square(P), 0) / 2.

			# first half-step
			P -= lf_step_sizes / 2. * dot(B, self.energy_gradient(Y + dot(B.T, Z)))
			Z += lf_step_sizes * P

			# full leapfrog steps
			for _ in range(lf_num_steps - 1):
				P -= lf_step_sizes * dot(B, self.energy_gradient(Y + dot(B.T, Z)))
				Z += lf_step_sizes * P

			# final half-step
			P -= lf_step_sizes / 2. * dot(B, self.energy_gradient(Y + dot(B.T, Z)))

			# new Hamiltonian
			Hnew = self.energy(Y + dot(B.T, Z)) + sum(square(P), 0) / 2.

			# Metropolis accept/reject step
			reject = (rand(1, Z.shape[1]) > exp(Hold - Hnew)).flatten()
			Z[:, reject] = Zold[:, reject]

			if adaptive:
				# adjust step sizes so that acceptance rate stays constant
				lf_step_sizes[:, reject] *= lf_step_size_dec
				lf_step_sizes[:,-reject] *= lf_step_size_inc

				# make sure step sizes don't become too small or large
				lf_step_sizes[lf_step_sizes > lf_step_size_max] = lf_step_size_max
				lf_step_sizes[lf_step_sizes < lf_step_size_min] = lf_step_size_min

			if Distribution.VERBOSITY > 1:
				print '{}\t{:.2f}\t{:.2f}\t{:.2f}'.format(step,
					mean(self.energy(Y + dot(B.T, Z))),
					mean(lf_step_sizes),
					mean(-reject))


		return Y + dot(B.T, Z)



	def sample_posterior_metropolis(self, X, num_steps=1000, Y=None, **kwargs):
		# hyperparameters
		acc_rate = kwargs.get('acc_rate', 0.9)
		adaptive = kwargs.get('adaptive', True)
		step_sizes = kwargs.get('step_size', 0.01) + zeros([1, X.shape[1]])
		step_size_min = kwargs.get('step_size_min', 0.001)
		step_size_max = kwargs.get('step_size_max', 0.2)
		step_size_dec = kwargs.get('step_size_dec', 0.95)
		step_size_inc = kwargs.get('step_size_inc', 
			1. / power(step_size_dec, (1. - acc_rate) / acc_rate))

		# nullspace basis
		B = svd(self.A)[2][self.num_visibles:, :]

		# initialization of nullspace hidden variables
		Z = dot(B, Y) if Y is not None else \
			dot(B, self.sample_prior(X.shape[1]))
		Y = dot(pinv(self.A), X)

		if Distribution.VERBOSITY > 1:
			print '{0:>6}{1:>10}{2:>10}{3:>10}'.format(
				'STEPS', 'ENERGY', 'ACC_RATE', 'STEP_SIZE')
			print '{0:6}{1:10.2f}{2:>10}{3:10.3f}'.format(0,
				mean(self.energy(Y + dot(B.T, Z))), '-', mean(step_sizes))

		for step in range(num_steps):
			Zold = copy(Z)
			Eold = self.energy(Y + dot(B.T, Z))

			Z += step_sizes * randn(*Z.shape)

			# new Hamiltonian
			Enew = self.energy(Y + dot(B.T, Z))

			# Metropolis accept/reject step
			reject = (log(rand(1, Z.shape[1])) > Eold - Enew).flatten()
			Z[:, reject] = Zold[:, reject]

			if adaptive:
				# adjust step sizes so that acceptance rate stays constant
				step_sizes[:, reject] *= step_size_dec
				step_sizes[:,-reject] *= step_size_inc

				# make sure step sizes don't become too small or large
				step_sizes[step_sizes > step_size_max] = step_size_max
				step_sizes[step_sizes < step_size_min] = step_size_min

			if Distribution.VERBOSITY > 1:
				print '{0:6}{1:10.2f}{2:10.2f}{3:10.3f}'.format(step + 1,
					mean(self.energy(Y + dot(B.T, Z))), 1. - mean(reject), mean(step_sizes))


		return Y + dot(B.T, Z)



	def energy_gradient(self, Y):
		gradient = []

		for model in self.subspaces:
			gradient.append(
				model.energy_gradient(Y[:model.dim]))
			Y = Y[model.dim:]

		return vstack(gradient)



	def energy(self, Y):
		energy = zeros([1, Y.shape[1]])

		for model in self.subspaces:
			energy += model.energy(Y[:model.dim])
			Y = Y[model.dim:]

		return energy
