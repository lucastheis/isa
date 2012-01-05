import sys
import unittest 

sys.path.append('./code')

from models import ICA
from numpy import zeros, all, abs

class Tests(unittest.TestCase):
	def test_prior_energy(self):
		step_size = 1E-5

		model = ICA(3)

		for mog in model.marginals:
			mog.initialize('student')

		# samples and true gradient
		X = model.sample_prior(100)
		G = model.prior_energy_gradient(X)

		# numerical gradient
		N = zeros(G.shape)
		for i in range(N.shape[0]):
			d = zeros(X.shape)
			d[i] = step_size
			N[i] = (model.prior_energy(X + d) - model.prior_energy(X - d)) / (2. * step_size)

		# test consistency of energy and gradient
		self.assertTrue(all(abs(G - N) < 1E-5))


	def test_training(self):
		ica = ICA(2)

		data = ica.sample(1000)

		# make sure SGD training doesn't throw any errors
		ica.train_sgd(data, max_iter=1)

		# make sure L-BFGS training doesn't throw any errors
		ica.train_lbfgs(data, max_fun=1)



if __name__ == '__main__':
	unittest.main()
