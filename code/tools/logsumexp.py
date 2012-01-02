"""
A numerically stable implementation of the logarithm of sums of exponentials.
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@theis.io>'
__docformat__ = 'epytext'

from numpy import log, sum, exp, zeros, max, asarray, vectorize, inf, nan, squeeze, reshape

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
