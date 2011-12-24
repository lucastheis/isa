__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@theis.io>'
__docformat__ = 'epytext'

from numpy import log, sum, exp, zeros, max, asarray, vectorize, inf, nan, squeeze, reshape
from scipy.special import gammainc
from scipy.optimize import bisect

def logsumexp(x, ax=None):
	"""
	Computes the log of the sum of the exp of the entries in x in a numerically
	stable way.

	@type  x: array_like
	@param x: a list, array or matrix of numbers

	@type  ax: integer
	@param ax: axis along which the sum is applied

	@rtype: array
	@return: an array containing the results
	"""

	if ax is None:
		x_max = max(x, ax)
		return x_max + log(sum(exp(x - x_max)))

	else:
		x_max_shape = list(x.shape)
		x_max_shape[ax] = 1

		x_max = asarray(max(x, ax))
		return x_max + log(sum(exp(x - x_max.reshape(x_max_shape)), ax))



def logmeanexp(x, ax=None):
	"""
	Computes the log of the mean of the exp of the entries in x in a numerically
	stable way. Uses logsumexp.

	@type  x: array_like
	@param x: a list, array or matrix of numbers

	@type  ax: integer
	@param ax: axis along which the values are averaged

	@rtype: array
	@return: an array containing the results
	"""

	x = asarray(x)
	n = x.size if ax is None else x.shape[ax]

	return logsumexp(x, ax) - log(n)



def gammaincinv(a, y, maxiter=100):
	"""
	A slower but more stable implementation of the inverse regularized
	incomplete Gamma function.
	"""

	y_min = 0.

	if y > 1:
		return nan

	# make sure range includes root
	while gammainc(a, gammaincinv.y_max) < y:
		y_min = gammaincinv.y_max
		gammaincinv.y_max += 1.

	# find inverse with bisection method
	return bisect(
	    f=lambda x: gammainc(a, x) - y,
	    a=y_min,
	    b=gammaincinv.y_max,
	    maxiter=maxiter,
	    xtol=1e-16,
	    disp=True)

gammaincinv = vectorize(gammaincinv)
gammaincinv.y_max = 1
