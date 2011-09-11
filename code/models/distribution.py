"""
Provides an interface which should be implemented by all probabilistic models.
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@bethgelab.org>'
__docformat__ = 'epytext'

from numpy import mean

class Distribution(object):
	"""
	Provides an interface and common functionality for probabilistic models.
	"""

	VERBOSITY = 2

	def __init__(self):
		raise NotImplementedError(str(self.__class__) + ' is an abstract class.')



	def sample(self, num_samples=1):
		"""
		Generate samples from the model.

		@type  num_samples: integer
		@param num_samples: the number of samples to generate
		"""

		raise NotImplementedError('Abstract method \'sample\' not implemented in '
			+ str(self.__class__))



	def initialize(self):
		"""
		Initializes the parameters of the distribution.
		"""

		raise NotImplementedError('Abstract method \'initialize\' not implemented in '
			+ str(self.__class__))



	def train(self, data, weights=None):
		"""
		Adapt the parameters of the model to the given set of data points.

		@type  data: array_like
		@param data: data stored in columns

		@type  weights: array_like
		@param weights: an optional weight for every data point
		"""

		raise NotImplementedError('Abstract method \'train\' not implemented in '
			+ str(self.__class__))



	def loglikelihood(self, data):
		"""
		Compute the log-likelihood of the model given the data.

		@type  data: array_like
		@param data: data stored in columns
		"""

		raise NotImplementedError('Abstract method \'loglikelihood\' not implemented in '
			+ str(self.__class__))



	def evaluate(self, data):
		"""
		Return average negative log-likelihood per dimension in nats.

		@type  data: array_like
		@param data: data stored in columns
		"""

		return -mean(self.loglikelihood(data)) / data.shape[0]



	def energy(self, data):
		return -self.loglikelihood(data)



	def energy_gradient(self, data):
		raise NotImplementedError('Abstract method \'energy_gradient\' not implemented in '
			+ str(self.__class__))
