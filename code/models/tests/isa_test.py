import sys
import unittest 

sys.path.append('./code')

from models import ISA, Distribution
from numpy import zeros, all, abs

Distribution.VERBOSITY = 0

class Tests(unittest.TestCase):
	def test_prior_energy(self):
		step_size = 1E-5

		model = ISA(3, 7, 1)

		for gsm in model.subspaces:
			gsm.initialize('student')

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



	def test_train(self):
		isa = ISA(2, 2)
		data = isa.sample(1000)

		# make sure SGD training doesn't throw any errors
		isa.train_sgd(data, max_iter=1)

		# make sure L-BFGS training doesn't throw any errors
		isa.train_lbfgs(data, max_fun=1)
	


	def test_train_subspaces(self):
		isa = ISA(4, 4, 2)
		isa.initialize(method='laplace')

		samples = isa.sample_prior(10000)

		isa = ISA(4, 4, 1)
		isa.initialize(method='laplace')

		isa.train_subspaces(samples, max_merge=5)
		isa.train_subspaces(samples, max_merge=5)

		self.assertTrue(len(isa.subspaces) == 2)



	def test_compute_map(self):
		isa = ISA(2, 4)

		X = isa.sample(100)

		M = isa.compute_map(X, tol=1E-4, maxiter=1000)
		Y = isa.sample_posterior(X)

		self.assertTrue(all(isa.prior_energy(M) <= isa.prior_energy(Y)))



if __name__ == '__main__':
	unittest.main()
