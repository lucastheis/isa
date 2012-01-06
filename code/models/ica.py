"""
An implementation of ICA using mixtures of Gaussian marginals.
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@theis.io>'
__docformat__ = 'epytext'

from numpy import *
from numpy.linalg import inv, det, slogdet
from numpy.random import *
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import laplace, t
from distribution import Distribution
from mogaussian import MoGaussian
from tools import mapp

class ICA(Distribution):
	def __init__(self, dim):
		self.dim = dim

		self.A = randn(dim, dim)

		self.marginals = [
			MoGaussian() for _ in range(dim)]

		for m in self.marginals:
			m.initialize()



	def initialize(self, method='laplace'):
		# fit mixture of Gaussian to Laplace
		mog = MoGaussian(num_components=self.marginals[0].num_components)

		if method.lower() == 'laplace':
			mog.train(laplace.rvs(size=[1, 10000]), max_iter=100)

		elif method.lower() == 'student':
			mog.train(t.rvs(1, size=[1, 10000]), max_iter=100)

		else:
			raise ValueError('Unknown initialization method \'{0}\'.'.format(method))

		for m in self.marginals:
			m.priors = mog.priors.copy()
			m.scales = mog.scales.copy()
			m.means = mog.means.copy()



	def train(self, X, method=('sgd', {}), **kwargs):
		max_iter = kwargs.get('max_iter', 100)
		adaptive = kwargs.get('adaptive', True) 
		train_prior = kwargs.get('train_prior', True)

		if isinstance(method, str):
			method = (method, {})

		if adaptive and 'step_width' not in method[1]:
			method[1]['step_width'] = 0.001

		if Distribution.VERBOSITY > 0:
			print 0, self.evaluate(X)

		for i in range(max_iter):
			# optimize parameters of the prior (M)
			if train_prior:
				self.train_prior(self.sample_posterior(X))

			# optimize linear features (M)
			if method[0].lower() == 'sgd':
				improved = self.train_sgd(X, **method[1])

				# adapt learning rate
				if adaptive:
					method[1]['step_width'] *= 1.1 if improved else 0.5

			elif method[0].lower() == 'lbfgs':
				self.train_lbfgs(X, **method[1])

			if Distribution.VERBOSITY > 0:
				print i + 1, self.evaluate(X)



	def train_prior(self, Y, **kwargs):
		def parfor(i):
			self.marginals[i].train(Y[[i]], **kwargs)
			self.marginals[i].normalize()
		mapp(parfor, range(self.dim))



	def train_sgd(self, X, **kwargs):
		# hyperparameters
		max_iter = kwargs.get('max_iter', 1)
		batch_size = kwargs.get('batch_size', min([100, X.shape[1]]))
		step_width = kwargs.get('step_width', 0.001)
		momentum = kwargs.get('momentum', 0.9)
		shuffle = kwargs.get('shuffle', True)
		pocket = kwargs.get('pocket', shuffle)

		# completed basis and filters
		A = self.A
		W = inv(A)

		# initial direction of momentum
		P = 0.

		if pocket:
			energy = mean(self.prior_energy(dot(W, X))) - slogdet(W)[1]

		for j in range(max_iter):
			if shuffle:
				# randomize order of data
				X = X[:, permutation(X.shape[1])]

			for i in range(0, X.shape[1], batch_size):
				batch = X[:, i:i + batch_size]

				if not batch.shape[1] < batch_size:
					# calculate gradient
					P = momentum * P + A.T - \
						dot(self.prior_energy_gradient(dot(W, batch)), batch.T) / batch_size

					# update parameters
					W += step_width * P
					A = inv(W)

		if pocket:
			# test for improvement of lower bound
			if mean(self.prior_energy(dot(W, X))) - slogdet(W)[1] > energy:
				if Distribution.VERBOSITY > 0:
					print 'No improvement.'

				# don't update parameters
				return False

		# update linear features
		self.A = A

		return True



	def train_lbfgs(self, X, **kwargs):
		# hyperparameters
		max_iter = kwargs.get('max_iter', 1)
		max_fun = kwargs.get('max_fun', 50)
		batch_size = kwargs.get('batch_size', X.shape[1])
		shuffle = kwargs.get('shuffle', True)
		pocket = kwargs.get('pocket', shuffle)

		# objective function
		def f(W, X):
			W = W.reshape(self.dim, self.dim)
			return mean(self.prior_energy(dot(W, X))) - slogdet(W)[1]

		# objective function gradient
		def df(W, X):
			W = W.reshape(self.dim, self.dim)
			A = inv(W)
			g = dot(self.prior_energy_gradient(dot(W, X)), X.T) / X.shape[1]
			return (g - A.T).flatten()

		# completed filter matrix
		W = inv(self.A)

		if pocket:
			energy = f(W, X)

		for _ in range(max_iter):
			if shuffle:
				# randomize order of data
				X = X[:, permutation(X.shape[1])]

			# split data in batches and perform L-BFGS on batches
			for i in range(0, X.shape[1], batch_size):
				batch = X[:, i:i + batch_size]

				if not batch.shape[1] < batch_size:
					W, _, _ = fmin_l_bfgs_b(f, W.flatten(), df, (batch,), maxfun=max_fun,
						disp=1 if Distribution.VERBOSITY > 1 else 0, iprint=0)

		if pocket:
			# test for improvement of lower bound
			if f(W, X) > energy:
				# don't update parameters
				return False

		# update linear features
		self.A = inv(W.reshape(*self.A.shape))

		return True



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
		@return: samples from the hidden marginal distributions
		"""

		return vstack(m.sample(num_samples) for m in self.marginals)



	def sample_posterior(self, X):
		return dot(inv(self.A), X) # faster than `solve` for large `X`

			
			
	def prior_energy_gradient(self, Y):
		"""
		Gradient of log-likelihood with respect to hidden state.
		"""

		energy_gradient = zeros([self.dim, Y.shape[1]])
		for i in range(self.dim):
			energy_gradient[i] = self.marginals[i].energy_gradient(Y[[i]])
		return energy_gradient



	def prior_energy(self, Y):
		"""
		For given hidden states, calculates the negative log-probability plus some constant.

		@type  Y: array_like
		@param Y: a number of hidden states stored in columns

		@rtype: ndarray
		@return: the negative log-porbability of each data point
		"""

		energy = zeros([1, Y.shape[1]])
		for i in range(self.dim):
			energy += self.marginals[i].energy(Y[[i]])
		return energy



	def prior_loglikelihood(self, Y):
		"""
		Calculates the log-probability of hidden states.

		@type  Y: array_like
		@param Y: a number of hidden states stored in columns

		@rtype: ndarray
		@return: the log-probability of each data point
		"""

		loglik = zeros([1, Y.shape[1]])
		for i in range(self.dim):
			loglik += self.marginals[i].loglikelihood(Y[[i]])
		return loglik



	def loglikelihood(self, X):
		return self.prior_loglikelihood(dot(inv(self.A), X)) - slogdet(self.A)[1]
