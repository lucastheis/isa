import sys
import unittest

sys.path.append('./code')

from transforms import WhiteningTransform
from numpy import cov, dot, all, eye, abs
from numpy.random import randn
from numpy.linalg import cholesky

class Tests(unittest.TestCase):
	def test_whitening(self):
		covr = cov(randn(16, 256))
		data = dot(cholesky(covr), randn(16, 10000))

		# test that whitening doesn't throw errors
		wt = WhiteningTransform(data, symmetric=True)
		
		# make sure data is white
		self.assertTrue(all(abs(cov(wt(data)) - eye(16)) < 1E-6))

		# make sure eigenvalues are sorted descendingly
		self.assertTrue(wt.eigvals[0] > wt.eigvals[-1])

		wt = WhiteningTransform(data, symmetric=False)

		# make sure data is white
		self.assertTrue(all(abs(cov(wt(data)) - eye(16)) < 1E-6))

		# make sure eigenvalues are sorted descendingly
		self.assertTrue(wt.eigvals[0] > wt.eigvals[-1])




if __name__ == '__main__':
	unittest.main()
