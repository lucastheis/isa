"""
Generates preprocessed datasets. 

	1. log-transforms and centers data
	2. adds Gaussian white noise
	3. removes DC component
	4. whitens data

"""

import sys

sys.path.append('./code')

from transforms import LinearTransform, WhiteningTransform
from numpy import load, sqrt, savez, cov
from tools import preprocess

def main(argv):
	patch_size = '8x8' if len(argv) < 2 else argv[1]

	# training data
	data_train = load('data/vanhateren.' + patch_size + '.1.npz')['data']
	data_train = preprocess(data_train, noise_level=32)

	# test data
	data_test = load('data/vanhateren.' + patch_size + '.0.npz')['data']
	data_test = preprocess(data_test, noise_level=32)

	# transforms
	dct = LinearTransform(dim=int(sqrt(data_train.shape[0])), basis='DCT')
	wt = WhiteningTransform(dct(data_train)[1:], symmetric=True)

	# further preprocess data
	data_train = wt(dct(data_train)[1:])
	data_test = wt(dct(data_test)[1:])

	savez('data/vanhateren.' + patch_size + '.preprocessed.npz',
		data_train=data_train,
		data_test=data_test,
		dct=dct.A,
		wt=wt.A)

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
