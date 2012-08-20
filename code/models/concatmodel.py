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
		self.models = list(args)



	def __len__(self):
		return len(self.models)



	def __getitem__(self, key):
		if isinstance(self.models, tuple):
			self.models = list(self.models)
		return self.models[key]



	def __setitem__(self, key, model):
		self.models[key] = model



	def train(self, data, model=None, **kwargs):
		"""
		Trains one or all models. If C{model} is `None`, all models are trained. If C{model} is an
		integer (starting at 0), only the specified model is trained. If C{model} is a list, the
		model with the index given by the first element is trained and the remainder of the list
		given as a parameter to that model.

		@type  model: int/list/None
		@param model: specifies which model to train
		"""

		if isinstance(model, list):
			if model:
				if model[1:]:
					kwargs['model'] = model[1:]
				model = model[0]
			else:
				model = None
		for i, m in enumerate(self.models):
			if model is None or model == i:
				m.train(data[:m.dim], **kwargs)
			data = data[m.dim:]



	def initialize(self, data=None, model=None, **kwargs):
		if isinstance(model, list):
			if model:
				if model[1:]:
					kwargs['model'] = model[1:]
				model = model[0]
			else:
				model = None
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
			loglik += m.loglikelihood(data[:m.dim], **kwargs)
			data = data[m.dim:]

		return loglik
