"""
An implementation of overcomplete ISA using Gaussian scale mixtures.
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@theis.io>'
__docformat__ = 'epytext'

from distribution import Distribution
from numpy import *
from numpy import min, max, round
from numpy.random import randint, randn, rand, logseries, permutation, gamma
from numpy.linalg import svd, pinv, inv, det, slogdet, cholesky, eig
from scipy.linalg import solve
from scipy.optimize import fmin_l_bfgs_b, fmin_cg, check_grad
from scipy.stats import laplace, t, cauchy, exponpow
from tools import gaborf, mapp, logmeanexp, asshmarray, sqrtmi, sqrtm
from warnings import warn
from gsm import GSM
from copy import deepcopy
from time import time

class ISA(Distribution):
	"""
	An implementation of overcomplete ISA using Gaussian scale mixtures.
	"""

	global_time = 0

	def __init__(self, num_visibles, num_hiddens=None, ssize=1, num_scales=10, noise=False):
		"""
		@type  num_visibles: integer
		@param num_visibles: data dimensionality

		@type  num_hiddens: integer
		@param num_hiddens: number of hidden units

		@type  ssize: integer
		@param ssize: subspace dimensionality

		@type  num_scales: integer
		@param num_scales: number of scales of each subspace GSM

		@type  noise: bool/ndarray
		@param noise: add additional hidden units for noise
		"""

		if num_hiddens is None:
			num_hiddens = num_visibles

		self.dim = num_visibles
		self.num_visibles = num_visibles
		self.num_hiddens = num_hiddens

		# random linear feature 
		self.A = randn(self.num_visibles, self.num_hiddens) / 10.

		# subspace densities
		self.subspaces = [
			GSM(ssize, num_scales) for _ in range(int(num_hiddens) / int(ssize))]

		if mod(num_hiddens, ssize) > 0:
			self.subspaces.append(GSM(mod(num_hiddens, ssize)))

		self._noise = False
		self.noise = noise



	def initialize(self, X=None, method='data'):
		"""
		Initializes parameter values with more sensible values.

		@type  X: array_like
		@param X: data points stored in columns

		@type  method: string
		@param method: type of initialization ('data', 'gabor' or 'random')
		"""

		if self.noise:
			L = self.A[:, :self.num_visibles]

		if method.lower() == 'data':
			# initialize features with data points
			if X is not None:
				if X.shape[1] < self.num_hiddens:
					raise ValueError('Number of data points to small.')

				else:
					# whitening matrix
					val, vec = eig(cov(X))

					# whiten data
					X_ = dot(dot(diag(1. / sqrt(val)), vec.T), X)

					# sort by norm in whitened space
					indices = argsort(sqrt(sum(square(X_), 0)))[::-1]

					# pick 25% largest data points and normalize
					X_ = X_[:, indices[:max([X.shape[1] / 4, self.num_hiddens])]]
					X_ = X_ / sqrt(sum(square(X_), 0))

					# pick first basis vector at random
					A = X_[:, [randint(X_.shape[1])]]

					for _ in range(self.num_hiddens - 1):
						# pick vector with large angle to all other vectors
						A = hstack([
							A, X_[:, [argmin(max(abs(dot(A.T, X_)), 0))]]])

					# orthogonalize and unwhiten
					A = dot(sqrtmi(dot(A, A.T)), A)
					A = dot(dot(vec, diag(sqrt(val))), A)

					self.A = A

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

		elif method.lower() in ['laplace', 'student', 'cauchy', 'exponpow']:
			if method.lower() == 'laplace':
				# approximate multivariate Laplace with GSM
				samples = randn(self.subspaces[0].dim, 10000)
				samples = samples / sqrt(sum(square(samples), 0))
				samples = laplace.rvs(size=[1, 10000]) * samples

			elif method.lower() == 'student':
				samples = randn(self.subspaces[0].dim, 50000)
				samples = samples / sqrt(sum(square(samples), 0))
				samples = t.rvs(2., size=[1, 50000]) * samples

			elif method.lower() == 'exponpow':
				exponent = 0.5
				samples = randn(self.subspaces[0].dim, 200000)
				samples = samples / sqrt(sum(square(samples), 0))
				samples = gamma(1. / exponent, 1., (1, 200000))**(1. / exponent) * samples

			else:
				samples = randn(self.subspaces[0].dim, 100000)
				samples = samples / sqrt(sum(square(samples), 0))
				samples = cauchy.rvs(size=[1, 100000]) * samples

			if self.noise:
				# ignore first subspace
				gsm = GSM(self.subspaces[1].dim, self.subspaces[1].num_scales)
				gsm.train(samples, max_iter=200, tol=1e-8)

				for m in self.subspaces[1:]:
					m.scales = gsm.scales.copy()
			else:
				# approximate distribution with GSM
				gsm = GSM(self.subspaces[0].dim, self.subspaces[0].num_scales)
				gsm.train(samples, max_iter=200, tol=1e-8)

				for m in self.subspaces:
					m.scales = gsm.scales.copy()

		else:
			raise ValueError('Unknown initialization method \'{0}\'.'.format(method))

		if self.noise:
			# don't initialize noise covariance
			self.A[:, :self.num_visibles] = L



	def train(self, X, method=('sgd', {}), sampling_method=('gibbs', {}), **kwargs):
		"""
		Optimizes model parameters with respect to the log-likelihoood. If the model is
		overcomplete, expectation maximization (EM) is used.

		The callback function takes two arguments: the model and the current iteration of the
		EM algorithm. It is called before the training starts and then after every iteration.

		@type  max_iter: integer
		@param max_iter: maximum number of iterations through the dataset

		@type  method: tuple
		@param method: optimization method used to optimize basis

		@type  adative: bool
		@param adaptive: automatically adjust step width when using SGD (default: True)

		@type  train_prior: bool
		@param train_prior: whether or not to optimize the marginal distributions (default: True)

		@type  train_subspaces: bool
		@param train_subspaces: automatically learn subspace sizes (default: False)

		@type  train_basis: bool
		@param train_basis: whether or not to optimize linear basis (default: True)

		@type  orthogonalize: bool
		@param orthogonalize: after each step, orthogonalize rows of feature matrix (default: False)

		@type  sampling_method: tuple
		@param sampling_method: method and parameters to generate hidden representations

		@type  persistent: bool
		@param persistent: initialize posterior samples with previous samples (default: True)

		@type  init_sampling_steps: integer
		@param init_sampling_steps: number of steps used to initialize persistent samples (default: 0)

		@type  callback: function
		@param callback: called after every iteration
		"""

		max_iter = kwargs.get('max_iter', 100)
		adaptive = kwargs.get('adaptive', True)
		train_prior = kwargs.get('train_prior', True)
		train_subspaces = kwargs.get('train_subspaces', False)
		train_basis = kwargs.get('train_basis', True)
		orthogonalize = kwargs.get('orthogonalize', False)
		persistent = kwargs.get('persistent', True)
		init_sampling_steps = kwargs.get('init_sampling_steps', 0)
		callback = kwargs.get('callback', None)

		if Distribution.VERBOSITY > 0:
			if self.num_hiddens > self.num_visibles:
				print 0
			else:
				print 0, self.evaluate(X)

		if isinstance(method, str):
			method = (method, {})

		if method[0].lower() == 'of':
			# don't sample, use sparse coding
			if callback:
				method[1]['callback'] = callback
			self.train_of(X, **method[1])
			return

		if callback:
			callback(self, 0)

		if isinstance(sampling_method, str):
			sampling_method = (sampling_method, {})

		if persistent and init_sampling_steps:
			# initialize samples
			sampling_method[1]['Z'] = self.sample_nullspace(X,
				method=(sampling_method[0], dict(sampling_method[1], num_steps=init_sampling_steps)))

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

			if train_basis and train_prior and (not orthogonalize):
				# normalize variances of marginals
				self.normalize_prior()

			if persistent:
				# initializes samples in next iteration
				sampling_method[1]['Z'] = dot(self.nullspace_basis(), Y)

			# optimize linear features (M)
			if train_basis:
				if method[0].lower() == 'analytic':
					self.train_analytic(Y, **method[1])

				elif method[0].lower() == 'sgd':
					improved = self.train_sgd(Y, **method[1])

					if adaptive:
						# adjust learning rate
						method[1]['step_width'] *= 1.1 if improved else 0.5

				elif method[0].lower() == 'lbfgs':
					self.train_lbfgs(Y, **method[1])

				else:
					raise ValueError('Unknown training method \'{0}\'.'.format(method[0]))

				if orthogonalize:
					# normalize feature matrix
					self.orthogonalize()

			if callback:
				callback(self, i + 1)

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
		@param Y: hidden stats
		"""

		max_iter = kwargs.get('max_iter', 10)
		tol = kwargs.get('tol', 1e-7)

		offset = [0]
		for model in self.subspaces:
			offset.append(offset[-1] + model.dim)

		def parfor(i):
			model = self.subspaces[i]
			model.train(Y[offset[i]:offset[i] + model.dim], max_iter=max_iter, tol=tol)
			return model
		self.subspaces = mapp(parfor, range(len(self.subspaces)))



	def normalize_prior(self):
		"""
		Normalizes the standard deviation of each subspace distribution.
		"""

		A = self.A

		for gsm in self.subspaces:
			# update basis
			A[:, :gsm.dim] *= gsm.std()
			A = A[:, gsm.dim:]

			# update marginals
			gsm.normalize()



	def train_subspaces(self, Y, **kwargs):
		"""
		Improves likelihood through spliting and merging of subspaces. This function
		may rearrange the order of subspaces and corresponding linear features.

		@type  max_merge: integer
		@param max_merge: maximum number of subspaces merged

		@type  max_iter: integer
		@param max_iter: maximum number of iterations for training joint L{GSM}

		@type  Y: array_like
		@param Y: hidden states

		@rtype: ndarray
		@return: data rearranged so that it aligns with subspaces
		"""

		max_merge = kwargs.get('max_merge', self.num_hiddens)
		max_iter = kwargs.get('max_iter', 10)

		if len(self.subspaces) > 1:
			# compute indices for each subspace
			indices = []
			index = 0
			for gsm in self.subspaces:
				indices.append(arange(gsm.dim) + index)
				index += gsm.dim

			# compute subspace energies
			energies = []
			for i, gsm in enumerate(self.subspaces):
				energies.append(sqrt(sum(square(Y[indices[i]]), 0)))
			energies = vstack(energies)

			# determine correlation of subspace energies
			corr = corrcoef(energies)
			corr = corr - triu(corr)

			if self._noise:
				# noise subspace shouldn't get merged
				corr[:, 0] = -1.

			for _ in range(max_merge):
				# pick subspaces with maximal correlation
				col = argmax(max(corr, 0))
				row = argmax(corr[:, col])

				if corr[row, col] <= 0.:
					break

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
					- self.subspaces[col].loglikelihood(Y_col) \
					- self.subspaces[row].loglikelihood(Y_row))

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



	def train_analytic(self, Y, **kwargs):
		"""
		Optimizes linear filters analytically. This only works if the model 
		has additive Gaussian noise enabled. Otherwise it won't change the filters.

		@type  train_noise: boolean
		@param train_noise: whether or not to update the noise covariance

		@rtype: bool
		@return: true if additive noise is enabled, otherwise false
		"""

		if not self.noise:
			warn('Wrong training method for noiseless model.')
			return False

		train_noise = kwargs.get('train_noise', True)

		# reconstruct data points
		X = dot(self.A, Y)

		# remove noise components
		Y = Y[self.num_visibles:, :]
		A = self.A[:, self.num_visibles:]

		# update basis
		Cyy = dot(Y, Y.T) / Y.shape[1]
		Cxy = dot(X, Y.T) / Y.shape[1]
		A = solve(Cyy, Cxy.T, sym_pos=True).T

		self.A[:, self.num_visibles:] = A

		if train_noise:
			# update covariance
			self.A[:, :self.num_visibles] = sqrtm(cov(X - dot(A, Y)))

		return True



	def train_lbfgs(self, Y, **kwargs):
		"""
		A stochastic variant of L-BFGS. If additive Gaussian noise is enabled, this method
		will always also optimize the covariance of the noise.

		@type  max_iter: integer
		@param max_iter: maximum number of iterations through data set

		@type  batch_size: integer
		@param batch_size: number of data points used for each L-BFGS update

		@type  pocket: bool
		@param pocket: if true, parameters are kept in case of performance degradation

		@rtype: bool
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
		weight_decay = kwargs.get('weight_decay', 0.)
		max_stored = kwargs.get('max_stored', 10)

		# objective function and gradient
		def f(W, X):
			W = W.reshape(self.num_hiddens, self.num_hiddens)
			v = mean(self.prior_energy(dot(W, X))) - slogdet(W)[1]

			if weight_decay > 0.:
				v += weight_decay / 2. * sum(square(inv(W)))
			return v

		# objective function gradient
		def df(W, X):
			W = W.reshape(self.num_hiddens, self.num_hiddens)
			A = inv(W)
			g = dot(self.prior_energy_gradient(dot(W, X)), X.T) / X.shape[1] - A.T

			if weight_decay > 0.:
				g -= weight_decay * dot(A.T, dot(A, A.T))

			return g.ravel()

		# complete basis
		A = vstack([self.A, self.nullspace_basis()])
		W = inv(A)

		# complete data
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
					W, _, _ = fmin_l_bfgs_b(f, W.ravel(), df, (batch,),
						maxfun=max_fun,
						m=max_stored,
						pgtol=1e-5,
						disp=1 if Distribution.VERBOSITY > 2 else 0,
						iprint=0)

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
		@param batch_size: number of data points used to approximate gradient (default: 100)

		@type  step_width: float
		@param step_width: factor by which gradient is multiplied (default: 0.001)

		@type  momentum: float
		@param momentum: fraction of previous parameter update added to gradient (default: 0.9)

		@type  shuffle: bool
		@param shuffle: before each iteration, randomize order of data (default: True)

		@type  pocket: bool
		@param pocket: do not update parameters in case of performance degradation (default: C{shuffle})

		@type  train_noise: boolean
		@param train_noise: whether or not to update the noise covariance

		@rtype: bool
		@return: false if no improvement could be achieved
		"""

		# hyperparameters
		max_iter = kwargs.get('max_iter', 1)
		batch_size = kwargs.get('batch_size', min([100, Y.shape[1]]))
		step_width = kwargs.get('step_width', 0.001)
		momentum = kwargs.get('momentum', 0.9)
		shuffle = kwargs.get('shuffle', True)
		pocket = kwargs.get('pocket', shuffle)
		train_noise = kwargs.get('train_noise', True)
		weight_decay = kwargs.get('weight_decay', 0.)
		natural_gradient = kwargs.get('natural_gradient', True)

		if self.noise:
			# reconstruct data points
			X = dot(self.A, Y)

			# remove noise components
			Y = Y[self.num_visibles:, :]
			A = self.A[:, self.num_visibles:]
			B = self.A[:, :self.num_visibles]
			S = inv(dot(B, B.T))

			# initial direction of momentum
			P = 0.

			if pocket:
				energy = mean(sqrt(sum(square(dot(inv(B), X - dot(A, Y))), 0))) - slogdet(S)[1] \
					+ weight_decay / 2. * sum(square(A))

			for _ in range(max_iter):
				if shuffle:
					# randomize order of data
					indices = permutation(X.shape[1])
					X = X[:, indices]
					Y = Y[:, indices]

				for i in range(0, X.shape[1], batch_size):
					# batches
					X_ = X[:, i:i + batch_size]
					Y_ = Y[:, i:i + batch_size]

					P = momentum * P + dot(S, dot(X_ - dot(A, Y_), Y_.T)) / batch_size

					if weight_decay > 0.:
						P -= weight_decay * dot(A.T, dot(A, A.T))

					A += step_width * P

			if train_noise:
				# estimate noise covariance
				C = cov(X - dot(A, Y))
				B = sqrtm(C)
			else:
				C = dot(B, B.T)

			if pocket:
				energy_ = mean(sqrt(sum(square(dot(inv(B), X - dot(A, Y))), 0))) + slogdet(C)[1] \
					+ weight_decay / 2. * sum(square(A))

				# test for improvement of lower bound
				if energy_ > energy:
					if Distribution.VERBOSITY > 0:
						print 'No improvement.'

					# don't update parameters
					return False

			if train_noise:
				self.A[:, :self.num_visibles] = B
			self.A[:, self.num_visibles:] = A

		else:
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
				energy = mean(self.prior_energy(Y)) - slogdet(W)[1] \
					+ weight_decay / 2. * sum(square(A))

			for j in range(max_iter):
				if shuffle:
					# randomize order of data
					X = X[:, permutation(X.shape[1])]

				for i in range(0, X.shape[1], batch_size):
					batch = X[:, i:i + batch_size]

					if not batch.shape[1] < batch_size:
						if natural_gradient:
							# calculate gradient
							P = momentum * P + W - \
								dot(dot(self.prior_energy_gradient(dot(W, batch)), batch.T) / batch_size, dot(W.T, W))

							# update parameters
							W += step_width * P
						else:
							# calculate gradient
							P = momentum * P + A.T - \
								dot(self.prior_energy_gradient(dot(W, batch)), batch.T) / batch_size

							if weight_decay > 0.:
								P -= weight_decay * dot(A.T, dot(A, A.T))

							# update parameters
							W += step_width * P
							A = inv(W)

			if natural_gradient:
				A = inv(W)

			if pocket:
				# test for improvement of lower bound
				energy_ = mean(self.prior_energy(dot(W, X))) - slogdet(W)[1] \
					+ weight_decay / 2. * sum(square(A))
				if energy_ > energy:
					if Distribution.VERBOSITY > 0:
						print 'No improvement.'

					# don't update parameters
					return False

			# update linear features
			self.A = A[:self.A.shape[0]]

		# TODO: will cause problems if adaptive = True and pocket = False
		return True



	def train_of(self, X, **kwargs):
		"""
		An implementation of Olshausen & Field's sparse coding algorithm.

		B{References:}
			- Olshausen, B. A. and Field, D. J. (1996). I{Emergence of simple-cell receptive field
			  properties by learning a sparse code for natural images}
		"""

		max_iter = kwargs.get('max_iter', 10)
		batch_size = kwargs.get('batch_size', 100)
		step_width = kwargs.get('step_width', 1.)
		momentum = kwargs.get('momentum', 0.)
		shuffle = kwargs.get('shuffle', True)
		noise_var = kwargs.get('noise_var', 0.005)  # noise variance
		alpha = kwargs.get('alpha', 0.02)
		var_eta = kwargs.get('var_eta', 0.01)
		var_goal = kwargs.get('var_goal', 0.1)
		beta = kwargs.get('beta', 1.2) # strength of the prior
		sigma = kwargs.get('sigma', 0.07) # dispersion of prior
		tol = kwargs.get('tol', 0.01)
		callback = kwargs.get('callback', None)

		if self.noise:
			# ignore model's noise covariance
			A = self.A[:, self.num_visibles:]
		else:
			A = self.A

		# estimated variance for each component
		Y_var = ones([1, A.shape[1]]) * var_goal
		gain = ones([1, A.shape[1]])

		# initial momentum
		P = 0.

		def compute_map(X):
			"""
			Computes the MAP for Laplacian prior and Gaussian additive noise.
			"""

			AA = dot(A.T, A)
			Ax = dot(A.T, X)

			def f(y, i):
				y = y.reshape(-1, 1)
				return sum(square(X[:, [i]] - dot(A, y))) / (2. * noise_var) + beta * sum(log(1. + square(y / sigma)))

			def df(y, i):
				y = y.reshape(-1, 1)
				grad = (dot(AA, y) - Ax[:, [i]]) / noise_var + (2. * beta / sigma**2) * y / (1. + square(y / sigma))
				return grad.ravel()

			# initial hidden states
			Y = asshmarray(dot(A.T, X) / sum(square(A), 0).reshape(-1, 1))

			def parfor(i):
				Y[:, i] = fmin_cg(f, Y[:, i], df, (i,), disp=False, maxiter=100, gtol=tol)
			mapp(parfor, range(X.shape[1]))

			return Y

		A = A / sqrt(sum(square(A), 0))

		if callback is not None:
			callback(self, 0)

		for i in range(max_iter):
			if shuffle:
				# randomize order of data
				X = X[:, permutation(X.shape[1])]

			for b in range(0, X.shape[1], batch_size):
				batch = X[:, b:b + batch_size]

				if not batch.shape[1] < batch_size:
					Y = compute_map(batch)

					# calculate gradient and update basis
					P = momentum * P + dot(batch - dot(A, Y), Y.T) / batch_size
					A += step_width * P

					# normalize basis
					Y_var = (1. - var_eta) * Y_var + var_eta * mean(square(Y), 1)
					gain *= power(Y_var / var_goal, alpha).reshape(1, -1)
					A = A / sqrt(sum(square(A), 0)) * gain

					if self.VERBOSITY > 0:
						print 'epoch {0}, batch {1}'.format(i, b / batch_size)
						print '{0:.4f} {1:.4f} {2:.4f}'.format(
							float(min(Y_var)), float(mean(Y_var)), float(max(Y_var)))

			if self.noise:
				self.A[:, self.num_visibles:] = A
			else:
				self.A = A

			if callback is not None:
				callback(self, i + 1)



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
			# repeat sampled scales for all subspace dimensions
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

		elif method[0].lower() == 'mala':
			return self.sample_posterior_mala(X, **method[1])

		elif method[0].lower() == 'metropolis':
			return self.sample_posterior_metropolis(X, **method[1])

		elif method[0].lower() == 'ais':
			return self.sample_posterior_ais(X, **method[1])[0]

		elif method[0].lower() == 'tempered':
			return self.sample_posterior_tempered(X, **method[1])

		else:
			raise ValueError('Unknown sampling method \'{0}\'.'.format(method))



	def sample_nullspace(self, X, method=('gibbs', {})):
		"""
		Draws a sample from the posterior over the linear model's null space.

		B{References:}
			- Chen, R. and Wu, Y. (2002). I{Null-Space Representation for Overcomplete Independent Component
			  Analysis}

		"""

		# nullspace basis
		return dot(self.nullspace_basis(), self.sample_posterior(X, method=method))



	def sample_posterior_gibbs(self, X, num_steps=10, Y=None, Z=None):
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
		if Z is None:
			Y = WX + dot(Q, Y) if Y is not None else \
				WX + dot(Q, self.sample_prior(X.shape[1]))
		else:
			V = pinv(self.nullspace_basis())
			Y = WX + dot(V, Z)

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

		# initialize proposal distribution to be Gaussian
		model = deepcopy(self)
		for gsm in model.subspaces:
			gsm.scales[:] = 1.

		# filter matrix and filter responses
		W = pinv(self.A)
		WX = dot(W, X)

		# nullspace basis and projection matrix
		B = self.nullspace_basis()
		Q = dot(B.T, B)

		# initialize proposal samples (Z is initially Gaussian and independent of X)
		Z = dot(B, randn(self.num_hiddens, X.shape[1]))
		Y = WX + dot(pinv(B), Z)

		# initialize importance weights (log-determinant of dot(B.T, B) not needed here)
		log_is_weights = sum(multiply(Z, dot(inv(dot(B, B.T)), Z)), 0) / 2. \
			+ (self.num_hiddens - self.num_visibles) / 2. * log(2. * pi)
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

		log_is_weights += self.prior_loglikelihood(Y) + slogdet(dot(W.T, W))[1] / 2.

		return Y, log_is_weights



	def sample_posterior_tempered(self, X, num_steps=1, annealing_weights=[], Y=None):
		"""
		Sample posterior distribution over hidden states using tempered transitions with
		Gibbs sampling transtition operator. This method might give better results if the
		marginals are very kurtotic and the posterior therefore highly multimodal.

		B{References:}
			- Neal, R. (1994). Sampling from Multimodal Distributions Using Tempered
			Transitions.
		"""

		if annealing_weights in ([], None):
			annealing_weights = linspace(0, 1, num_steps + 1)[1:]

		# initialize distribution
		model = deepcopy(self)

		# filter matrix and filter responses
		W = pinv(self.A)
		WX = dot(W, X)

		# nullspace basis and projection matrix
		B = self.nullspace_basis()
		Q = dot(B.T, B)

		# initial hidden state
		Y = WX + dot(Q, Y) if Y is not None else \
			WX + dot(Q, self.sample_prior(X.shape[1]))

		for _ in range(num_steps):
			Y_old = copy(Y)

			# initialize importance weights
			log_is_weights = self.prior_energy(Y)

			# increase temperature
			for step, beta in enumerate(annealing_weights[::-1]):
				# tune proposal distribution
				for i in range(len(self.subspaces)):
					# adjust standard deviations
					model.subspaces[i].scales = (1. - beta) + beta * self.subspaces[i].scales

				log_is_weights -= model.prior_energy(Y)

				# apply transition operator
				S = model.sample_scales(Y)
				Y = model._sample_posterior_cond(Y, X, S, W, WX, Q)

				log_is_weights += model.prior_energy(Y)

				if Distribution.VERBOSITY > 1:
					print '{0:6}\t{1:10.2f}'.format(step + 1, mean(self.prior_energy(Y)))

			# decrease temperature
			for step, beta in enumerate(annealing_weights):
				# tune proposal distribution
				for i in range(len(self.subspaces)):
					# adjust standard deviations
					model.subspaces[i].scales = (1. - beta) + beta * self.subspaces[i].scales

				log_is_weights -= model.prior_energy(Y)

				# apply transition operator
				S = model.sample_scales(Y)
				Y = model._sample_posterior_cond(Y, X, S, W, WX, Q)

				log_is_weights += model.prior_energy(Y)

				if Distribution.VERBOSITY > 1:
					print '{0:6}\t{1:10.2f}'.format(len(annealing_weights) - step, mean(self.prior_energy(Y)))

			log_is_weights -= self.prior_energy(Y)

			# Metropolis accept/reject step
			reject = (rand(1, X.shape[1]) > exp(log_is_weights)).ravel()
			Y[:, reject] = Y_old[:, reject]

			if Distribution.VERBOSITY > 1:
				print mean(-reject), 'accepted'

		return Y



	def _sample_posterior_cond(self, Y, X, S, W, WX, Q):
		"""
		Samples posterior conditioned on scales.

		B{References:}
			- Doucet, A. (2010). I{A Note on Efficient Conditional Simulation of
			Gaussian Distributions.}
		"""

		# sample hidden states conditioned on scales
		Y_ = multiply(randn(self.num_hiddens, X.shape[1]), S)

		X_ = X - dot(self.A, Y_)

		# variances and incomplete covariance matrices
		v = square(S).reshape(-1, 1, X.shape[1])
		C = multiply(v, self.A.T.reshape(self.num_hiddens, -1, 1)).transpose([2, 0, 1]) # TODO: FIX MEMORY ISSUES

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

		@type  lf_num_steps: integer
		@param lf_num_steps: number of leap-frog steps (default: 10)

		@type  lf_step_size: float
		@param lf_step_size: leap-frog step size (default: 0.01)

		@type  lf_randomness: float
		@param lf_randomness: relative jitter added to step size (default: 0.)

		B{References:}
			- Duane, A. (1987). I{Hybrid Monte Carlo.}
			- Neal, R. (2010). I{MCMC Using Hamiltonian Dynamics.}
		"""

		# hyperparameters
		lf_num_steps = kwargs.get('lf_num_steps', 10)
		lf_step_size = kwargs.get('lf_step_size', 0.01)
		lf_randomness = kwargs.get('lf_randomness', 0.)

		# nullspace basis and projection matrix
		B = self.nullspace_basis()
		BB = dot(B.T, B)

		# filter responses
		WX = dot(pinv(self.A), X)

		# initial hidden state
		Y = WX + dot(BB, Y) if Y is not None else \
			WX + dot(BB, self.sample_prior(X.shape[1]))

		for step in range(num_steps):
			lf_step_size_rnd = (1. + lf_randomness * (2. * rand() - 1.)) * lf_step_size

			# sample momentum
			P = randn(B.shape[0], X.shape[1])

			# store Hamiltonian
			Yold = copy(Y)
			Hold = self.prior_energy(Y) + sum(square(P), 0) / 2.

			# first half-step
			P -= lf_step_size_rnd / 2. * dot(B, self.prior_energy_gradient(Y))
			Y += lf_step_size_rnd * dot(B.T, P)

			# full leapfrog steps
			for _ in range(lf_num_steps - 1):
				P -= lf_step_size_rnd * dot(B, self.prior_energy_gradient(Y))
				Y += lf_step_size_rnd * dot(B.T, P)

			# final half-step
			P -= lf_step_size_rnd / 2. * dot(B, self.prior_energy_gradient(Y))

			# make sure hidden and visible states stay consistent
			Y = WX + dot(BB, Y)

			# new Hamiltonian
			Hnew = self.prior_energy(Y) + sum(square(P), 0) / 2.

			# Metropolis accept/reject step
			reject = (rand(1, X.shape[1]) > exp(Hold - Hnew)).ravel()
			Y[:, reject] = Yold[:, reject]

			if Distribution.VERBOSITY > 1:
				print '{0:6}\t{1:10.2f}\t{2:10.2f}'.format(step + 1,
					mean(self.prior_energy(Y)),
					mean(-reject))

		return Y



	def sample_posterior_mala(self, X, num_steps=100, Y=None, **kwargs):
		"""
		This is a special case of HMC sampling.
		"""

		step_width = kwargs.get('step_width', 0.01)

		# nullspace basis and projection matrix
		B = self.nullspace_basis()
		BB = dot(B.T, B)

		# filter responses
		WX = dot(pinv(self.A), X)

		# initial hidden state
		Y = WX + dot(BB, Y) if Y is not None else \
			WX + dot(BB, self.sample_prior(X.shape[1]))

		for step in range(num_steps):
			P = randn(B.shape[0], X.shape[1])

			# store Hamiltonian
			Yold = Y
			Hold = self.prior_energy(Y) + sum(square(P), 0) / 2.

			# generate proposal sample
			P = P - step_width / 2. * dot(B, self.prior_energy_gradient(Y))
			Y = WX + dot(BB, Y) + step_width * dot(B.T, P)
			P = P - step_width / 2. * dot(B, self.prior_energy_gradient(Y))

			# new Hamiltonian
			Hnew = self.prior_energy(Y) + sum(square(P), 0) / 2.

			# Metropolis accept/reject step
			reject = (rand(1, X.shape[1]) > exp(Hold - Hnew)).ravel()
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
			reject = (log(rand(1, Z.shape[1])) > Eold - Enew).ravel()
			Z[:, reject] = Zold[:, reject]

			if Distribution.VERBOSITY > 1:
				print '{0:6}{1:10.2f}{2:10.2f}'.format(step + 1,
					mean(self.prior_energy(WX + dot(B.T, Z))), 1. - mean(reject))

		return WX + dot(B.T, Z)



	def compute_map(self, X, tol=1E-3, maxiter=1000):
		"""
		Try to find the MAP of the posterior using conjugate gradient descent.
		If the posterior is multimodal, a local optimum will be found.
		"""

		W = pinv(self.A)
		V = svd(W)[0][:, self.num_visibles:]

		WX = dot(W, X)

		def f(z, i):
			return self.prior_energy(WX[:, [i]] + dot(V, z.reshape(-1, 1))).ravel()

		def df(z, i):
			return dot(V.T, self.prior_energy_gradient(WX[:, [i]] + dot(V, z.reshape(-1, 1)))).ravel()

		# initial nullspace state
		Z = asshmarray(zeros([self.num_hiddens - self.num_visibles, X.shape[1]]))

		def parfor(i):
			Z[:, i] = fmin_cg(f, Z[:, i], df, (i,), disp=False, maxiter=maxiter, gtol=tol)
		map(parfor, range(X.shape[1]))

		return WX + dot(V, Z)
			
			

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



	def loglikelihood(self, X, num_samples=10, method='biased', sampling_method=('ais', {'num_steps': 10}), **kwargs):
		"""
		Computes the log-likelihood (in nats) for a set of data samples. If the model is overcomplete,
		the log-likelihood is estimated using one of two importance sampling methods. The biased method
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

		@type  return_all: boolean
		@param return_all: if true, return all important weights and don't average (default: False)

		@rtype: ndarray
		@return: the log-probability of each data point
		"""

		return_all = kwargs.get('return_all', False)

		if self.num_hiddens == self.num_visibles:
			return self.prior_loglikelihood(dot(inv(self.A), X)) - slogdet(self.A)[1]

		else:
			if method == 'biased':
				# sample importance weights
				log_is_weights = asshmarray(empty([num_samples, X.shape[1]]))
				def parfor(i):
					log_is_weights[i] = self.sample_posterior_ais(X, **sampling_method[1])[1]
				mapp(parfor, range(num_samples))

				if return_all:
					return asarray(log_is_weights)
				else:
					# average importance weights to get log-likelihoods
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

				if return_all:
					return loglik
				else:
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

		@rtype: bool
		@return: whether or not the model is overcomplete
		"""

		return self.num_hiddens > self.num_visibles



	def orthogonalize(self):
		"""
		Symmetrically orthogonalizes the rows of the feature matrix. If additive Gaussian
		noise is enabled, the columns representing the noise will not be affected.
		"""

		if self.noise:
			A = self.A[:, self.num_visibles:]
			self.A[:, self.num_visibles:] = dot(sqrtmi(dot(A, A.T)), A)
		else:
			self.A = dot(sqrtmi(dot(self.A, self.A.T)), self.A)

	

	@property
	def noise(self):
		"""
		Whether or not noise is explicitly modeled.
		"""

		return self._noise



	@noise.setter
	def noise(self, noise):
		"""
		Enables or disables additive Gaussian noise. Disabling the noise will
		delete the stored noise covariance matrix.

		@type  noise: ndarray/bool
		@param noise: the covariance matrix of the assumed noise or True/False
		"""

		if isinstance(noise, ndarray):
			if not self._noise:
				self._noise = True

				# add Gaussian subspace representing noise
				self.subspaces.insert(0, GSM(self.num_visibles, 1))
				self.A = hstack([eye(self.num_visibles) / 20., self.A])
				self.num_hiddens += self.num_visibles

			self.A[:, :self.num_visibles] = sqrtm(noise)

		else:
			if self._noise != noise:
				self._noise = noise
				
				if self._noise:
					# add Gaussian subspace representing noise
					self.subspaces.insert(0, GSM(self.num_visibles, 1))
					self.A = hstack([eye(self.num_visibles) / 20., self.A])
					self.num_hiddens += self.num_visibles

				else:
					# remove subspace representing noise
					self.subspaces.remove(0)
					self.A = self.A[:, self.num_visibles:]
					self.num_hiddens -= self.num_visibles
