import sys

sys.path.append('./code')

from models import ICA, ISA
from numpy import load
from tools import preprocess

def main(argv):
	data = load('data/vanhateren.8x8.1.npz')['data']
	data = preprocess(data, noise_level=32)

	data_train = data[:, :50000]
	data_test = data[:, 50000:100000]

	ica = ISA(data.shape[0], ssize=1) 
	ica.train(data_train, max_iter=10, method='sgd', train_prior=False)
	ica.train(data_train, max_iter=40, method='sgd', train_prior=True)
	ica.train(data_train, max_iter=10, method='lbfgs', train_prior=True)

	print ica.evaluate(data_test)

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
