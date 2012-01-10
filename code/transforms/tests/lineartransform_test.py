import sys
import unittest

sys.path.append('./code')

from transforms import LinearTransform
from numpy import dot, eye, max, abs, all
from numpy.random import randn

class Tests(unittest.TestCase):
	def test_dct(self):
		dct = LinearTransform(dim=16, basis='DCT')

		# make sure DCT basis is orthogonal
		self.assertTrue(all(abs(dot(dct.A, dct.A.T) - eye(256)) < 1E-10))
		self.assertTrue(all(abs(dct.logjacobian(randn(16, 10))) < 1E-10))



if __name__ == '__main__':
	unittest.main()
