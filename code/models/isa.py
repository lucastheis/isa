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
from numpy.linalg import svd, pinv, inv, det, slogdet
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
		self.A = randn(self.num_visibles, self.num_hiddens) / sqrt(self.num_hiddens)

		# subspace densities
		self.subspaces = [
			GSM(ssize) for i in range(num_hiddens / ssize)]

		# initialize subspace distributions
		for model in self.subspaces:
			model.initialize()



	def initialize(self, X=None, method='gabor'):
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
				# center data
				self.A = X[:, randint(X.shape[1], size=self.num_hiddens)] / sqrt(self.num_hiddens)
				self.A /= sqrt(sum(square(self.A), 0))

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

		elif method.lower() == 'random':
			# initialize with Gaussian white noise
			self.A = randn(num_visibles, num_hiddens)

		else:
			raise ValueError('Unknown initialization method \'{0}\'.'.format(method))



	def train(self, X, max_iter=100, method=('sgd', {}), sampling_method=('gibbs', {})):
		"""
		@type  max_iter: integer
		@param max_iter: maximum number of iterations through the dataset

		@type  method: tuple
		@param method: optimization method used to optimize filters

		@type  sampling_method: tuple
		@param sampling_method: method to generate hidden representations
		"""

		if Distribution.VERBOSITY > 0:
			print 0, self.evaluate(X) / log(2.)

		for i in range(max_iter):
			# complete data (E)
			Y = self.sample_posterior(X, method=sampling_method)

			# optimize parameters with respect to completed data (M)
			if i >= 30:
				self.train_prior(Y)
			self.train_sgd(Y, **method[1])

			if Distribution.VERBOSITY > 0:
				print i + 1, self.evaluate(X) / log(2.)



	def train_prior(self, Y, **kwargs):
		for model in self.subspaces:
			model.train(Y[:model.dim])
			model.normalize()
			Y = Y[model.dim:]



	def train_sgd(self, Y, **kwargs):
		max_iter = kwargs.get('max_iter', 1)
		batch_size = kwargs.get('batch_size', 100)
		step_width = kwargs.get('step_width', 0.001)
		momentum = kwargs.get('momentum', 0.8)

		# nullspace basis
		B = svd(self.A)[2][self.num_visibles:, :]
		
		# completed basis and filters
		A = vstack([self.A, B])
		W = inv(A)

		# completed data
		XZ = dot(A, Y)

		P = 0.

		for _ in range(max_iter):
			for i in range(0, XZ.shape[1], batch_size):
				batch = XZ[:, i:i + batch_size]

				if not batch.shape[1] < batch_size:
					# calculate gradient
					P = momentum * P + A.T - \
						dot(self.prior_energy_gradient(dot(W, batch)), batch.T) / batch_size

					# update parameters
					W += step_width * P
					A = inv(W)

		# update model parameters
		self.A = A[:self.A.shape[0]]



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
			return dot(inv(self.A), X) # faster than `solve` for large `X`

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

			if Distribution.VERBOSITY > 1:
				print '{0:6}\t{1:10.2f}'.format(step + 1, mean(self.prior_energy(Y)))

		return asarray(Y)



	def sample_posterior_hmc(self, X, num_steps=100, Y=None, **kwargs):
		# hyperparameters
		lf_num_steps = kwargs.get('lf_num_steps', 10)
		lf_step_size = kwargs.get('lf_step_size', 0.01) + zeros([1, X.shape[1]])

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
			Hold = self.prior_energy(Y + dot(B.T, Z)) + sum(square(P), 0) / 2.

			# first half-step
			P -= lf_step_size / 2. * dot(B, self.prior_energy_gradient(Y + dot(B.T, Z)))
			Z += lf_step_size * P

			# full leapfrog steps
			for _ in range(lf_num_steps - 1):
				P -= lf_step_size * dot(B, self.prior_energy_gradient(Y + dot(B.T, Z)))
				Z += lf_step_size * P

			# final half-step
			P -= lf_step_size / 2. * dot(B, self.prior_energy_gradient(Y + dot(B.T, Z)))

			# new Hamiltonian
			Hnew = self.prior_energy(Y + dot(B.T, Z)) + sum(square(P), 0) / 2.

			# Metropolis accept/reject step
			reject = (rand(1, Z.shape[1]) > exp(Hold - Hnew)).flatten()
			Z[:, reject] = Zold[:, reject]

			if Distribution.VERBOSITY > 1:
				print '{0:6}\t{1:10.2f}\t{2:10.2f}'.format(step + 1,
					mean(self.prior_energy(Y + dot(B.T, Z))),
					mean(-reject))

		return Y + dot(B.T, Z)



	def sample_posterior_metropolis(self, X, num_steps=1000, Y=None, **kwargs):
		# hyperparameters
		standard_deviation = kwargs.get('standard_deviation', 0.01)

		# nullspace basis
		B = svd(self.A)[2][self.num_visibles:, :]

		# initialization of nullspace hidden variables
		Z = dot(B, Y) if Y is not None else \
			dot(B, self.sample_prior(X.shape[1]))
		Y = dot(pinv(self.A), X)

		for step in range(num_steps):
			Zold = copy(Z)
			Eold = self.prior_energy(Y + dot(B.T, Z))

			Z += standard_deviation * randn(*Z.shape)

			# new Hamiltonian
			Enew = self.prior_energy(Y + dot(B.T, Z))

			# Metropolis accept/reject step
			reject = (log(rand(1, Z.shape[1])) > Eold - Enew).flatten()
			Z[:, reject] = Zold[:, reject]

			if Distribution.VERBOSITY > 1:
				print '{0:6}{1:10.2f}{2:10.2f}'.format(step + 1,
					mean(self.prior_energy(Y + dot(B.T, Z))), 1. - mean(reject))

		return Y + dot(B.T, Z)



	def prior_energy_gradient(self, Y):
		"""
		Gradient of log-likelihood with respect to hidden state.
		"""

		gradient = []

		for model in self.subspaces:
			gradient.append(
				model.energy_gradient(Y[:model.dim]))
			Y = Y[model.dim:]

		return vstack(gradient)



	def prior_energy(self, Y):
		"""
		For given hidden states, calculates the negative log-probability plus some constant.
		"""

		energy = zeros([1, Y.shape[1]])

		for model in self.subspaces:
			energy += model.energy(Y[:model.dim])
			Y = Y[model.dim:]

		return energy



	def prior_loglikelihood(self, Y):
		"""
		Calculates the log-probability of hidden states.
		"""

		energy = zeros([1, Y.shape[1]])

		for model in self.subspaces:
			energy += model.loglikelihood(Y[:model.dim])
			Y = Y[model.dim:]

		return energy



	def loglikelihood(self, X):
		if self.num_hiddens == self.num_visibles:
			W = inv(self.A)
			return (-mean(self.prior_loglikelihood(dot(W, X))) - slogdet(W)[1]) / X.shape[0]
		else:
			raise NotImplementedError()
