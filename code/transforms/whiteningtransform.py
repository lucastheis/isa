__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@theis.io>'
__docformat__ = 'epytext'

from lineartransform import LinearTransform
from numpy import asarray, mean, cov, real, dot, argsort, diag, sqrt
from numpy.linalg import eig

class WhiteningTransform(LinearTransform):
	def __init__(self, data, symmetric=True, tol=1E-10):
		"""
		If C{symmetric} is true, symmetric/ZCA whitening is performed. Otherwise, PCA
		whitening is performed such that the returned features with the lowest indices
		correspond to the components with the largest variance.

		Directions of very small variance will be ignored, so that a
		low-dimensional manifold stays a low-dimensional manifold after
		transformation.

		@type  data: array_like
		@param data: data used to compute covariance matrix

		@type  symmetric: boolean
		@param symmetric: if true, perform symmetric whitening

		@type  tol: float
		@param tol: directions with eigenvalues smaller than C{tol} get ignored
		"""

		self.symmetric = symmetric

		data = asarray(data, dtype='float64')

		# center
		data = data - mean(data, 1).reshape(-1, 1)

		# find eigenvectors
		eigvals, eigvecs = eig(cov(data))
		eigvals, eigvecs = real(eigvals), real(eigvecs)

		# sort eigenvectors
		indices = argsort(eigvals)[::-1]
		eigvals = eigvals[indices]
		eigvecs = eigvecs[:, indices]

		self.eigvals = eigvals

		if symmetric:
			# eliminate eigenvectors whose eigenvalues are zero
			eigvecs = eigvecs[:, eigvals > tol]
			eigvals = eigvals[eigvals > tol]

			# symmetric whitening matrix
			whitening_matrix = dot(eigvecs, dot(diag(1. / sqrt(eigvals)), eigvecs.T))

		else:
			# ignore directions of very small variance
			eigvals[eigvals <= tol] = 1.

			# whitening matrix
			whitening_matrix = dot(eigvecs, dot(diag(1. / sqrt(eigvals)), eigvecs.T))

		LinearTransform.__init__(self, whitening_matrix)
