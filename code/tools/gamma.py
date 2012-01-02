__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@theis.io>'
__docformat__ = 'epytext'

from numpy import vectorize, nan
from scipy.special import gammainc
from scipy.optimize import bisect

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
