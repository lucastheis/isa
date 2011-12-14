"""
Tool for preprocessing data in a standard way.
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@tuebingen.mpg.de>'
__docformat__ = 'epytext'
__version__ = '1.2.0'

from numpy import log, transpose, mean, dot, diag, sqrt, cov, asarray, real, std
from numpy.random import permutation, randn
from numpy.linalg import eig

def preprocess(data, return_whitening_matrix=False, etol=1E-10, noise_level=None):
	"""
	Log-transforms, centers and symmetrically whitens data.

	@type  data: array_like
	@param data: data points stored in columns

	@type  etol: float
	@param etol: eigenvalues below this threshold are not considered

	@type  noise_level: integer
	@param noise_level: add a little bit of noise after log-transform

	@rtype: ndarray/tuple
	@return: preprocessed data, optionally with whitening matrix
	"""

	data = asarray(data, dtype='float64')

	# log-transform
	data[data < 1.] = 1.
	data = log(data)

	# center
	data = data - mean(data, 1).reshape(-1, 1)

	if noise_level is not None:
		# add Gaussian white noise
		data += randn(*data.shape) * (std(data) / float(noise_level))

	# shuffle
	data = data[:, permutation(data.shape[1])]

	# find eigenvectors
	eigvals, eigvecs = eig(cov(data))
	eigvals, eigvecs = real(eigvals), real(eigvecs)

	# eliminate eigenvectors whose eigenvalues are zero
	eigvecs = eigvecs[:, eigvals > etol]
	eigvals = eigvals[eigvals > etol]

	# symmetric whitening matrix
	whitening_matrix = dot(eigvecs, dot(diag(1. / sqrt(eigvals)), eigvecs.T))

	# whiten data
	if return_whitening_matrix:
		return asarray(dot(whitening_matrix, data), order='F'), whitening_matrix
	else:
		return asarray(dot(whitening_matrix, data), order='F')



def whiten(data, return_whitening_matrix=False, etol=1E-10):
	"""
	Symmetric whitening.

	@type  data: array_like
	@param data: data points stored in columns

	@type  etol: float
	@param etol: eigenvalues below this threshold are not considered

	@rtype: ndarray/tuple
	@return: whitened data, optionally with whitening matrix
	"""

	data = asarray(data, dtype='float64')

	# center
	data = data - mean(data, 1).reshape(-1, 1)

	# find eigenvectors
	eigvals, eigvecs = eig(cov(data))
	eigvals, eigvecs = real(eigvals), real(eigvecs)

	# eliminate eigenvectors whose eigenvalues are zero
	print min(eigvals)
	eigvecs = eigvecs[:, eigvals > etol]
	eigvals = eigvals[eigvals > etol]

	# symmetric whitening matrix
	whitening_matrix = dot(eigvecs, dot(diag(1. / sqrt(eigvals)), eigvecs.T))

	# whiten data
	if return_whitening_matrix:
		return asarray(dot(whitening_matrix, data), order='F'), whitening_matrix
	else:
		return asarray(dot(whitening_matrix, data), order='F')
