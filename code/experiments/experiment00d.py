import sys

sys.path.append('./code')

from models import ISA, ICA, GSM
from transforms import RadialGaussianization
from numpy import load
from tools import Experiment, preprocess, mapp

mapp.max_processes = 1

def main(argv):
	experiment = Experiment()

	data = load('data/vanhateren.8x8.1.npz')['data']
	data = preprocess(data, noise_level=32)

	gsm = GSM(data.shape[0], 10)
	gsm.train(data[:50000])

	data = RadialGaussianization(gsm).apply(data)

	ica = ICA(data.shape[0])

	ica.initialize(method='laplace')
	ica.train(data[:20000], max_iter=20, method='sgd', train_prior=False)
	ica.train(data[:50000], max_iter=5, method='lbfgs')

	experiment['gsm'] = gsm
	experiment['ica'] = ica
	experiment.save('results/experiment00a/experiment00a.{0}.{1}.xpck')

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
