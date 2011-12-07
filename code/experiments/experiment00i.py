import sys

sys.path.append('./code')

from natter import DataModule
from natter.Distributions import CompleteLinearModel, ProductOfExponentialPowerDistributions
from natter.Transforms import LinearTransformFactory
from tools import preprocess
from numpy import load

def main(argv):
	data = load('data/vanhateren.8x8.1.npz')['data']
	data = preprocess(data)

	data_train = DataModule.Data(data[:, :50000])
	data_test = DataModule.Data(data[:, 50000:100000])

	W = LinearTransformFactory.fastICA(data_train)

	ica = CompleteLinearModel({
		'n': data.shape[0], 
		'W': W,
		'q': ProductOfExponentialPowerDistributions({'n': data.shape[0]})})
	ica.estimate(data_train)

	print ica.all(data_test)

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
