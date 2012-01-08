"""
An implementation of overcomplete ISA using Gaussian scale mixtures.
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@theis.io>'
__docformat__ = 'epytext'

from distribution import Distribution
from numpy import *
from numpy import min, max, round
from numpy.random import randint, randn, rand, logseries, permutation
from numpy.linalg import svd, pinv, inv, det, slogdet
from scipy.linalg import solve
from scipy.optimize import fmin_l_bfgs_b, check_grad
from scipy.stats import laplace
from tools import gaborf, mapp, whiten, logmeanexp, asshmarray
from gsm import GSM
from copy import deepcopy

class ISA(Distribution):
	"""
	An implementation of overcomplete ISA using Gaussian scale mixtures.
	"""

	def __init__(self, num_visibles, num_hiddens=None, ssize=1):
		"""
		@type  num_visibles: integer
		@param num_visibles: data dimensionality

		@type  num_hiddens: integer
		@param num_hiddens: number of hidden units

		@type  ssize: integer
		@param ssize: subspace dimensionality
		"""

		if num_hiddens is None:
			num_hiddens = num_visibles

		if mod(num_hiddens, ssize):
			raise ValueError('num_hiddens must be a multiple of ssize.')

		self.num_visibles = num_visibles
		self.num_hiddens = num_hiddens

		# random linear feature 
		self.A = randn(self.num_visibles, self.num_hiddens) / 10.

		# subspace densities
		self.subspaces = [
			GSM(ssize) for _ in range(num_hiddens / ssize)]

		# initialize subspace distributions
		for model in self.subspaces:
			model.initialize()



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

		elif method.lower() == 'laplace':
			# fit mixture of Gaussian to multivariate Laplace
			samples = randn(self.subspaces[0].dim, 10000)
			samples = samples / sqrt(sum(square(samples), 0))
			samples = laplace.rvs(size=[1, 10000]) * samples

			gsm = GSM(self.subspaces[0].dim, self.subspaces[0].num_scales)
			gsm.train(samples, max_iter=100)

			for m in self.subspaces:
				m.scales = gsm.scales.copy()

		else:
			raise ValueError('Unknown initialization method \'{0}\'.'.format(method))



	def train(self, X, method=('sgd', {}), sampling_method=('gibbs', {}), **kwargs):
		"""
		Optimizes model parameters with respect to the log-likelihoood.

		@type  max_iter: integer
		@param max_iter: maximum number of iterations through the dataset

		@type  method: tuple
		@param method: optimization method used to optimize filters

		@type  train_prior: boolean
		@param train_prior: whether or not to optimize the marginal distributions

		@type  sampling_method: tuple
		@param sampling_method: method and parameters to generate hidden representations
		"""

		max_iter = kwargs.get('max_iter', 100)
		adaptive = kwargs.get('adaptive', True) 
		train_prior = kwargs.get('train_prior', True)
		train_subspaces = kwargs.get('train_subspaces', False)

		if Distribution.VERBOSITY > 0:
			if self.num_hiddens > self.num_visibles:
				print 0
			else:
				print 0, self.evaluate(X)

		if isinstance(method, str):
			method = (method, {})

		if isinstance(sampling_method, str):
			sampling_method = (sampling_method, {})

		if adaptive and 'step_width' not in method[1]:
			method[1]['step_width'] = 0.001

		for i in range(max_iter):
			# complete data (E)
			Y = self.sample_posterior(X, method=sampling_method)

			if train_prior:
				# optimize parameters of the prior (M)
				self.train_prior(Y)

			if train_subspaces:
				# learn subspaces (M)
				Y = self.train_subspaces(Y)

			# used to initialize sampling procedure in next iteration
			sampling_method[1]['Y'] = Y

			# optimize linear features (M)
			if method[0].lower() == 'sgd':
				improved = self.train_sgd(Y, **method[1])

				if adaptive:
					# adjust learning rate
					method[1]['step_width'] *= 1.1 if improved else 0.5

			elif method[0].lower() == 'lbfgs':
				self.train_lbfgs(Y, **method[1])

			else:
				raise ValueError('Unknown training method \'{0}\'.'.format(method[0]))

			if Distribution.VERBOSITY > 0:
				if self.num_hiddens > self.num_visibles:
					print i + 1
				else:
					print i + 1, self.evaluate(X)



	def train_prior(self, Y, **kwargs):
		"""
		Optimize parameters of the marginal distribution over the hidden variables.
		The parameters are fit to maximize the average log-likelihood of the
		columns in `Y`.

		@type  Y: array_like
		@param Y: hidden states
		"""

		offset = [0]
		for model in self.subspaces:
			offset.append(offset[-1] + model.dim)

		def parfor(i):
			model = self.subspaces[i]
			model.train(Y[offset[i]:offset[i] + model.dim])
			model.normalize()
			return model

		self.subspaces = mapp(parfor, range(len(self.subspaces)))



	def train_subspaces(self, Y, **kwargs):
		"""
		Improves likelihood through spliting and merging of subspaces. This function
		may rearrange the order of subspaces and corresponding linear features.

		@type  max_merge: integer
		@param max_merge: maximum number of subspaces merged

		@type  Y: array_like
		@param Y: hidden states

		@rtype: ndarray
		@return: data rearranged so that it aligns with subspaces
		"""

		max_merge = kwargs.get('max_merge', 10)
		max_iter = kwargs.get('max_iter', 10)

		# calculate indices for each subspace
		indices = []
		index = 0
		for gsm in self.subspaces:
			indices.append(arange(gsm.dim) + index)
			index += gsm.dim

		# calculate subspace energies
		energies = []
		for i, gsm in enumerate(self.subspaces):
			energies.append(sqrt(sum(square(Y[indices[i]]), 0)))
		energies = vstack(energies)

		# determine correlation of subspace energies
		corr = corrcoef(energies)
		corr = corr - triu(corr)

		for _ in range(max_merge):
			# pick subspaces with maximal correlation
			col = argmax(max(corr, 0))
			row = argmax(corr[:, col])

			if corr[row, col] <= 0.:
				break

			# make sure subspace combination won't be selected again
			corr[row, col] = 0.

			# extract data from subspaces
			Y_row = Y[indices[row]]
			Y_col = Y[indices[col]]
			Y_jnt = vstack([Y_row, Y_col])

			# train joint model
			gsm = GSM(Y_jnt.shape[0], self.subspaces[col].num_scales)
			gsm.scales = self.subspaces[col].scales.copy()
			gsm.train(Y_jnt, max_iter=max_iter)

			# log-likelihood improvement
			mi = mean(gsm.loglikelihood(Y_jnt) \
				- self.subspaces[col].loglikelihood(Y_row) \
				- self.subspaces[row].loglikelihood(Y_col))

			if mi > 0:
				self.subspaces.append(gsm)

				# rearrange linear filters
				subspace_indices = concatenate([indices[row], indices[col]])
				self.A = hstack([self.A, self.A[:, subspace_indices]])
				self.A = delete(self.A, subspace_indices, 1)

				# rearrange data
				Y = vstack([Y, Y[subspace_indices, :]])
				Y = delete(Y, subspace_indices, 0)

				# remove subspaces from correlation matrix
				corr = delete(corr, [row, col], 0)
				corr = delete(corr, [row, col], 1)

				# update indices
				for k in range(row + 1, len(indices)):
					indices[k] -= self.subspaces[row].dim
				for k in range(col + 1, len(indices)):
					indices[k] -= self.subspaces[col].dim

				if row < col:
					del self.subspaces[col]
					del self.subspaces[row]
					del indices[col]
					del indices[row]
				else:
					del self.subspaces[row]
					del self.subspaces[col]
					del indices[row]
					del indices[col]

				if Distribution.VERBOSITY > 0:
					print 'Merged subspaces.'

				if corr.size == 0:
					break

			return Y



	def train_lbfgs(self, Y, **kwargs):
		"""
		A stochastic variant of L-BFGS.

		@type  max_iter: integer
		@param max_iter: maximum number of iterations through data set

		@type  batch_size: integer
		@param batch_size: number of data points used for each L-BFGS update

		@type  step_width: float
		@param step_width: factor by which gradient is multiplied

		@type  pocket: boolean
		@param pocket: if true, parameters are kept in case of performance degradation

		@rtype: boolean
		@return: false if no improvement could be achieved

		B{References}:
			- Byrd, R. H., Lu, P. and Nocedal, J. (1995). I{A Limited Memory Algorithm for Bound
			Constrained Optimization.}
		"""

		# hyperparameters
		max_iter = kwargs.get('max_iter', 1)
		max_fun = kwargs.get('max_fun', 50)
		batch_size = kwargs.get('batch_size', Y.shape[1])
		shuffle = kwargs.get('shuffle', True)
		pocket = kwargs.get('pocket', shuffle)

		# objective function
		def f(W, X):
			W = W.reshape(self.num_hiddens, self.num_hiddens)
			return mean(self.prior_energy(dot(W, X))) - slogdet(W)[1]

		# objective function gradient
		def df(W, X):
			W = W.reshape(self.num_hiddens, self.num_hiddens)
			A = inv(W)
			g = dot(self.prior_energy_gradient(dot(W, X)), X.T) / X.shape[1]
			return (g - A.T).flatten()

		# completed filter matrix
		A = vstack([self.A, self.nullspace_basis()])
		W = inv(A)

		# completed data
		X = dot(A, Y)

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
		self.A = inv(W.reshape(*A.shape))[:self.A.shape[0]]

		return True



	def train_sgd(self, Y, **kwargs):
		"""
		Optimize linear features to maximize the joint log-likelihood of visible
		and nullspace states.

		@type  max_iter: integer
		@param max_iter: maximum number of iterations through data set

		@type  batch_size: integer
		@param batch_size: number of data points used to approximate gradient

		@type  step_width: float
		@param step_width: factor by which gradient is multiplied

		@type  momentum: float
		@param momentum: fraction of previous parameter update added to gradient

		@type  pocket: boolean
		@param pocket: if true, parameters are kept in case of performance degradation

		@rtype: boolean
		@return: false if no improvement could be achieved
		"""

		# hyperparameters
		max_iter = kwargs.get('max_iter', 1)
		batch_size = kwargs.get('batch_size', min([100, Y.shape[1]]))
		step_width = kwargs.get('step_width', 0.001)
		momentum = kwargs.get('momentum', 0.9)
		shuffle = kwargs.get('shuffle', True)
		pocket = kwargs.get('pocket', shuffle)

		# nullspace basis
		B = self.nullspace_basis()
		
		# completed basis and filters
		A = vstack([self.A, B])
		W = inv(A)

		# completed data
		X = dot(A, Y)

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
		self.A = A[:self.A.shape[0]]

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

		elif method[0].lower() == 'ais':
			return self.sample_posterior_ais(X, **method[1])[0]

		else:
			raise ValueError('Unknown sampling method \'{0}\'.'.format(method))



	def sample_nullspace(self, X, method=('gibbs', {})):
		"""
		Draws a sample from the posterior over the linear model's null space.
		"""

		# nullspace basis
		return dot(self.nullspace_basis(), self.sample_posterior(X, method=method))



	def sample_posterior_gibbs(self, X, num_steps=10, Y=None):
		"""
		B{References:}
			- Doucet, A. (2010). I{A Note on Efficient Conditional Simulation of
			Gaussian Distributions.}
		"""

		# filter matrix and filter responses
		W = pinv(self.A)
		WX = dot(W, X)

		# nullspace projection matrix
		Q = eye(self.num_hiddens) - dot(W, self.A)

		# initial hidden state
		Y = WX + dot(Q, Y) if Y is not None else \
			WX + dot(Q, self.sample_prior(X.shape[1]))

		# Gibbs sample between S and Y given X
		for step in range(num_steps):
			# update scales
			S = self.sample_scales(Y)

			# update hidden states
			Y = self._sample_posterior_cond(Y, X, S, W, WX, Q)

			if Distribution.VERBOSITY > 1:
				print '{0:6}\t{1:10.2f}'.format(step + 1, mean(self.prior_energy(Y)))

		return asarray(Y)



	def sample_posterior_ais(self, X, num_steps=10, annealing_weights=[]):
		"""
		Sample posterior distribution over hidden states using annealed importance
		sampling with Gibbs sampling transition operator.
		"""

		if not annealing_weights:
			annealing_weights = linspace(0, 1, num_steps + 1)[1:]

		# initialize proposal distribution to Gaussian
		model = deepcopy(self)
		for gsm in model.subspaces:
			gsm.scales[:] = 1.

		# filter matrix and filter responses
		W = pinv(self.A)
		WX = dot(W, X)

		# nullspace basis and projection matrix
		B = self.nullspace_basis()
		Q = dot(B.T, B)

		# initialize proposal samples (X and Z are initially independent and Gaussian)
		Z = dot(B, randn(self.num_hiddens, X.shape[1]))
		Y = WX + dot(pinv(B), Z)

		# initialize importance weights
		log_is_weights = sum(multiply(Z, dot(inv(dot(B, B.T)), Z)), 0) / 2. \
			+ (self.num_hiddens - self.num_visibles) / 2. * log(2. * pi) + slogdet(dot(W.T, W))[1] / 2.
		log_is_weights.resize(1, X.shape[1])

		for step, beta in enumerate(annealing_weights):
			# tune proposal distribution
			for i in range(len(self.subspaces)):
				# adjust standard deviations
				model.subspaces[i].scales = (1. - beta) + beta * self.subspaces[i].scales

			log_is_weights -= model.prior_energy(Y)

			# apply Gibbs sampling transition operator
			S = model.sample_scales(Y)
			Y = model._sample_posterior_cond(Y, X, S, W, WX, Q)

			log_is_weights += model.prior_energy(Y)

			if Distribution.VERBOSITY > 1:
				print '{0:6}\t{1:10.2f}'.format(step + 1, mean(self.prior_energy(Y)))

		log_is_weights += self.prior_loglikelihood(Y)

		return Y, log_is_weights
			


	def _sample_posterior_cond(self, Y, X, S, W, WX, Q):
		"""
		Samples posterior conditioned on scales. Ugly, but efficient.

		B{References:}
			- Doucet, A. (2010). I{A Note on Efficient Conditional Simulation of
			Gaussian Distributions.}
		"""

		# sample hidden states conditioned on scales
		Y_ = multiply(randn(self.num_hiddens, X.shape[1]), S)

		X_ = X - dot(self.A, Y_)

		# variances and incomplete covariance matrices
		v = square(S).reshape(-1, 1, X.shape[1])
		C = multiply(v, self.A.T.reshape(self.num_hiddens, -1, 1)).transpose([2, 0, 1])

		# update hidden states
		Y = asshmarray(Y)
		def parfor(i):
			Y[:, i] = dot(C[i], solve(dot(self.A, C[i]), X_[:, i], sym_pos=True))
		mapp(parfor, range(X.shape[1]))

		return WX + dot(Q, Y + Y_)



	def sample_posterior_hmc(self, X, num_steps=100, Y=None, **kwargs):
		"""
		Samples posterior over hidden representations using Hamiltonian Monte
		Carlo sampling.

		B{References:}
			- Duane, A. (1987). I{Hybrid Monte Carlo.}
			- Neal, R. (2010). I{MCMC Using Hamiltonian Dynamics}
		"""

		# hyperparameters
		lf_num_steps = kwargs.get('lf_num_steps', 10)
		lf_step_size = kwargs.get('lf_step_size', 0.01) + zeros([1, X.shape[1]])

		# nullspace basis
		B = self.nullspace_basis()

		if Y is None:
			# initialize hidden variables
			Y = self.sample_prior(X.shape[1])

		# make sure hidden and visible states are consistent
		WX = dot(pinv(self.A), X)
		BB = dot(B.T, B)
		Y = Wx + dot(BB, Y)

		for step in range(num_steps):
			# sample momentum
			P = randn(B.shape[0], X.shape[1])

			# store Hamiltonian
			Yold = copy(Y)
			Hold = self.prior_energy(Y) + sum(square(P), 0) / 2.

			# first half-step
			P -= lf_step_size / 2. * dot(B, self.prior_energy_gradient(Y))
			Y += lf_step_size * dot(B.T, P)

			# full leapfrog steps
			for _ in range(lf_num_steps - 1):
				P -= lf_step_size * dot(B, self.prior_energy_gradient(Y))
				Y += lf_step_size * dot(B.T, P)

			# final half-step
			P -= lf_step_size / 2. * dot(B, self.prior_energy_gradient(Y))

			# make sure hidden and visible state stay consistent
			Y = WX + dot(BB, Y)

			# new Hamiltonian
			Hnew = self.prior_energy(Y) + sum(square(P), 0) / 2.

			# Metropolis accept/reject step
			reject = (rand(1, X.shape[1]) > exp(Hold - Hnew)).flatten()
			Y[:, reject] = Yold[:, reject]

			if Distribution.VERBOSITY > 1:
				print '{0:6}\t{1:10.2f}\t{2:10.2f}'.format(step + 1,
					mean(self.prior_energy(Y)),
					mean(-reject))

		return Y



	def sample_posterior_metropolis(self, X, num_steps=1000, Y=None, **kwargs):
		"""
		Sample posterior over hidden representations using Metropolis-Hastings
		"""

		# hyperparameters
		standard_deviation = kwargs.get('standard_deviation', 0.01)

		# nullspace basis
		B = self.nullspace_basis()

		# initialization of nullspace hidden variables
		Z = dot(B, Y) if Y is not None else \
			dot(B, self.sample_prior(X.shape[1]))

		WX = dot(pinv(self.A), X)

		for step in range(num_steps):
			Zold = copy(Z)
			Eold = self.prior_energy(WX + dot(B.T, Z))

			Z += standard_deviation * randn(*Z.shape)

			# new Hamiltonian
			Enew = self.prior_energy(WX + dot(B.T, Z))

			# Metropolis accept/reject step
			reject = (log(rand(1, Z.shape[1])) > Eold - Enew).flatten()
			Z[:, reject] = Zold[:, reject]

			if Distribution.VERBOSITY > 1:
				print '{0:6}{1:10.2f}{2:10.2f}'.format(step + 1,
					mean(self.prior_energy(WX + dot(B.T, Z))), 1. - mean(reject))

		return WX + dot(B.T, Z)
			
			

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

		@type  Y: array_like
		@param Y: a number of hidden states stored in columns

		@rtype: ndarray
		@return: the negative log-porbability of each data point
		"""

		energy = zeros([1, Y.shape[1]])

		for model in self.subspaces:
			energy += model.energy(Y[:model.dim])
			Y = Y[model.dim:]

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

		for model in self.subspaces:
			loglik += model.loglikelihood(Y[:model.dim])
			Y = Y[model.dim:]

		return loglik



	def loglikelihood(self, X, num_samples=10, method='biased', sampling_method=('ais', {'num_steps': 10})):
		"""
		Computes the log-likelihood for a set of data samples. If the model is overcomplete, the
		log-likelihood is estimated using one of two importance sampling methods. The biased method
		tends to underestimate the log-likelihood. To get rid of the bias, use more samples.
		The unbiased method oftentimes suffers from extremely high variance and should be used with
		caution.

		@type  X: array_like
		@param X: a number of visible states stored in columns

		@type  method: string
		@param method: whether to use the 'biased' or 'unbiased' method

		@type  num_samples: integer
		@param num_samples: number of generated importance weights

		@type  sampling_method: tuple
		@param sampling_method: method and parameters to generate importance weights

		@rtype: ndarray
		@return: the log-probability of each data point
		"""

		if self.num_hiddens == self.num_visibles:
			return self.prior_loglikelihood(dot(inv(self.A), X)) - slogdet(self.A)[1]

		else:
			if method == 'biased':
				# sample importance weights
				log_is_weights = asshmarray(empty([num_samples, X.shape[1]]))
				def parfor(i):
					log_is_weights[i] = self.sample_posterior_ais(X, **sampling_method[1])[1]
				mapp(parfor, range(num_samples))

				# average importance weights to get likelihoods
				return logmeanexp(log_is_weights, 0)

			elif method == 'unbiased':
				loglik = empty(X.shape[1])

				# sample importance weights
				log_is_weights = asshmarray(empty([num_samples, X.shape[1]]))
				def parfor(i):
					log_is_weights[i] = self.sample_posterior_ais(X, **sampling_method[1])[1]
				mapp(parfor, range(num_samples))

				# obtain an initial first guess using the biased method
				is_weights = exp(log_is_weights)
				is_mean = mean(is_weights, 0)
				is_var = var(is_weights, 0, ddof=1)

				# Taylor series expansion points
				c = (is_var + square(is_mean)) / is_mean

				# logarithmic series distribution parameters
				p = sqrt(is_var / (is_var + square(is_mean)))

				# sample "number of importance samples" for each data point
				num_samples = array([logseries(p_) for p_ in p], dtype='uint32')

				for k in range(1, max(num_samples) + 1):
					# data points for which to generate k importance weights
					indices = where(num_samples == k)[0]

					# sample importance weights
					if len(indices) > 0:
						log_is_weights = asshmarray(empty([k, len(indices)]))

						def parfor(i):
							log_is_weights[i] = self.sample_posterior_ais(X[:, indices], num_steps=num_steps)[1]
						mapp(parfor, range(k))

						# hyperparameter used for selected datapoints
						c_ = c[indices]
						p_ = p[indices]

						# unbiased estimate of log-likelihood
						loglik[indices] = log(c_) + log(1. - p_) * prod((c_ - exp(log_is_weights)) / (c_ * p_), 0)

				return mean(loglik, 0).reshape(1, -1)

			else:
				raise NotImplementedError('Unknown method \'{0}\'.'.format(method))



	def nullspace_basis(self):
		"""
		Compute the orthogonal complement of the feature matrix.
		"""

		return svd(self.A)[2][self.num_visibles:, :]



	def is_overcomplete(self):
		"""
		Return C{true} if model is overcomplete, otherwise C{false}.

		@rtype: boolean
		@return: whether or not the model is overcomplete
		"""

		return self.num_hiddens > self.num_visibles
