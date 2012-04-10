"""
Allows you to model certain dimensions of data separately.
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@theis.io>'
__docformat__ = 'epytext'

from distribution import Distribution
from numpy import vstack, sum, zeros

class ConcatModel(Distribution):
	def __init__(self, *args):
		self.dim = sum(m.dim for m in args)
		self.models = args



	def __len__(self):
		return len(self.models)



	def __getitem__(self, key):
		return self.models[key]



	def train(self, data, model=None, **kwargs):
		for i, m in enumerate(self.models):
			if model is None or model == i:
				m.train(data[:m.dim], **kwargs)
			data = data[m.dim:]



	def initialize(self, data=None, model=None, **kwargs):
		if data is None:
			for i, m in enumerate(self.models):
				if model is None or model == i:
					m.initialize(**kwargs)
		else:
			for i, m in enumerate(self.models):
				if model is None or model == i:
					m.initialize(data[:m.dim], **kwargs)
				data = data[m.dim:]



	def sample(self, num_samples=1):
		return vstack(m.sample(num_samples) for m in self.models)



	def loglikelihood(self, data, **kwargs):
		loglik = zeros([1, data.shape[1]])

		for m in self.models:
			loglik = loglik + m.loglikelihood(data[:m.dim], **kwargs)
			data = data[m.dim:]

		return loglik
