"""
An implementation of hierarchical overcomplete ISA.
"""

class HISA(Distribution):
	def __init__(self, num_visibles=None, training_data=None, test_data=None):
		self.training_data = [training_data]
		self.test_data = [test_data]

		self.layers = []

		# determine number of visible units
		if training_data is not None:
			self.num_visibles = training_data.shape[0]
		elif test_data is not None:
			self.num_visibles = test_data.shape[0]
		else:
			if num_visibles is None:
				raise ValueError('Please specify the number of visible units.')
			self.num_visibles = num_visibles



	def __len__(self):
		return len(self.layers)



	def __getitem__(self, key):
		return self.layers[key]



	def add_layer(self, num_hiddens):
		"""
		Adds a layer to the top of the hierarchical model.
		"""

		if self.layers:
			self.layers.append(ISA(self.layers[-1].num_hiddens, num_hiddens))
		else:
			self.layers.append(self.num_visibles, num_hiddens)



	def train(self, data=None, max_iter=100, method=('sgd', {}), sampling_method=('gibbs', {})):
		"""
		Train the top most layer.
		"""

		self.generate_data(data)

		if len(self.training_data) < len(self):
			raise ValueError('No training data.')

		self[-1].train(self.training_data[-1], 
			max_iter=max_iter, 
			method=method, 
			sampling_method=sampling_method)



	def generate_data(self, training_data=None, test_data=None, sampling_method=('gibbs', {})):
		"""
		Creates training and test data for the top most layer.
		"""

		if training_data is not None:
			self.training_data = [training_data]

		if test_data is not None:
			self.test_data = [test_data]

		if self.training_data:
			if Distribution.VERBOSITY > 0:
				print 'generating training data...'

			while len(self.training_data) < len(self):
				self.training_data.append(
					self.layers[len(self.training_data) - 1].sample_posterior(
						self.training_data[-1]), method=sampling_method)
				self.training_data[-1] = \
					self.layers[len(self.training_data) - 1].gaussianize(
						self.training_data[-1])

		if self.test_data:
			if Distribution.VERBOSITY > 0:
				print 'generating test data...'

			while len(self.test_data) < len(self):
				self.test_data.append(
					self.layers[len(self.test_data) - 1].sample_posterior(
						self.test_data[-1]), method=sampling_method)
				self.test_data[-1] = \
					self.layers[len(self.test_data) - 1].gaussianize(
						self.test_data[-1])



	def loglikelihood(self):
		pass



	def evaluate(self):
		pass
