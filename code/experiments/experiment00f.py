"""
Train natter implementation of Lp-spherical distribution on image patches.
"""

import sys

sys.path.append('./code')

from natter import DataModule
from natter.Distributions import LpSphericallySymmetric, CompleteLinearModel
from natter.Transforms import LinearTransformFactory
from tools import preprocess
from numpy import load, eye, cov
from numpy.random import randn

def main(argv):
	data = load('data/vanhateren.16x16.preprocessed.npz')

	data_train = DataModule.Data(data['data_train'])
	data_test = DataModule.Data(data['data_test'])

	dim = data['data_train'].shape[0]

	W = LinearTransformFactory.fastICA(data_train)

	p = CompleteLinearModel({
		'n': dim, 
		'W': W,
		'q': LpSphericallySymmetric(n=data['data_train'].shape[0], p=1.5)})
	p.estimate(data_train)

	print p.all(data_test)

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
