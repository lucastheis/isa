__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@theis.io>'
__docformat__ = 'epytext'

from scipy.special import erf, erfinv
from scipy.stats import norm
from scipy.optimize import bisect
from numpy import mean, sqrt, asarray
from transforms import Transform

class UnivariateGaussianization(Transform):
	def __init__(self, mog):
		self.mog = mog



	def apply(self, data):
		# make sure data has right shape
		data = asarray(data).reshape(1, -1)

		# apply model CDF
		data = self.mog.cdf(data)

		# apply inverse Gaussian CDF
		return erfinv(data * 2. - 1.) * sqrt(2.)



	def inverse(self, data, max_iter=100):
		# make sure data has right shape
		data = asarray(data).reshape(1, -1)

		# apply Gaussian CDF
		data = norm.cdf(data)

		# apply inverse model CDF
		val_max = mean(self.mog.means) + 1.
		val_min = mean(self.mog.means) - 1.

		for t in range(data.shape[1]):
			# make sure root lies between val_min and val_max
			while float(self.mog.cdf(val_min)) > data[0, t]:
				val_min -= 1.
			while float(self.mog.cdf(val_max)) < data[0, t]:
				val_max += 1.

			# find root numerically
			data[0, t] = bisect(
			    f=lambda x: float(self.mog.cdf(x)) - data[0, t],
			    a=val_min,
			    b=val_max,
			    maxiter=max_iter,
			    disp=False)

		return data


	
	def logjacobian(self, data):
		# make sure data has right shape
		data = asarray(data).reshape(1, -1)
		return self.mog.loglikelihood(data) - norm.logpdf(self.apply(data))
