"""
Matrix square root and inverse matrix square root.
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@theis.io>'
__docformat__ = 'epytext'
__version__ = '1.0.0'

from numpy import log, mean, dot, diag, sqrt
from numpy.linalg import eig

def sqrtm(mat):
	"""
	Matrix square root.

	@type  mat: array_like
	@param mat: matrix for which to compute square root
	"""

	# find eigenvectors
	eigvals, eigvecs = eig(mat)

	# matrix square root
	return dot(eigvecs, dot(diag(sqrt(eigvals)), eigvecs.T))



def sqrtmi(mat):
	"""
	Compute matrix inverse square root.

	@type  mat: array_like
	@param mat: matrix for which to compute inverse square root
	"""

	# find eigenvectors
	eigvals, eigvecs = eig(mat)

	# eliminate eigenvectors whose eigenvalues are zero
	eigvecs = eigvecs[:, eigvals > 0]
	eigvals = eigvals[eigvals > 0]

	# inverse square root
	return dot(eigvecs, dot(diag(1. / sqrt(eigvals)), eigvecs.T))
