import sys

sys.path.append('./code')

from tools import Experiment, preprocess
from numpy import *
from numpy import sort
from numpy.linalg import inv
from pgf import *
from transforms import RadialGaussianization
from isa import GSM

def main(argv):
	### MARGINAL GAUSSIANIZATION

	results = Experiment('results/c_vanhateren/c_vanhateren.12.25072012.003003.xpck')

	ica = results['model'].model[1].model

	t = linspace(-4, 4, 200)

	cyclelist('fruity')

	for gsm in ica.subspaces()[1:4]:
		rg = RadialGaussianization(gsm)
		y = rg(t.reshape(1, -1))

		plot(t, y, line_width=2)

	title('Marginal Gaussianization')
	ylabel('$\phi(y_i)$')
	xlabel('$y_i$')
	gcf().sans_serif = True

	savefig('marginal_gaussianization.tex')

	### RADIAL GAUSSIANIZATION

	figure()

	cyclelist('fruity')

	results = Experiment('results/c_hisa/c_hisa.0.1.xpck')

	isa = results['model'].model[1].model

	patch_size = int(sqrt(results['model'].dim) + 0.5)
	data_train = load('data/vanhateren.{0}x{0}.0.npz'.format(patch_size))['data']
	data_train = preprocess(data_train, shuffle=False)[:, :100000]
	dct = results['model'].transforms[0]
	wt = results['model'].model[1].transforms[0]

	states = dot(inv(isa.A), wt(dct(data_train)[1:]))
	off = 0

	for gsm in isa.subspaces()[:3]:
		# finetune GSM
		gsm = GSM(gsm.dim, 50)
		gsm.train(states[off:off + gsm.dim], tol=1e-9, max_iter=100)

		norms = sqrt(sum(square(states[off:off + gsm.dim]), 0))

		t = linspace(percentile(norms, 1), percentile(norms, 99), 200)

		rg = RadialGaussianization(gsm)
		u = ones([gsm.dim, t.size]) / sqrt(gsm.dim)
		y = sqrt(sum(square(rg(u * t.reshape(1, -1))), 0))

		plot(t, y, line_width=2)

		off += gsm.dim

	title('Radial Gaussianization')
	xlabel('$||y_I||$')
	ylabel('$\phi(||y_I||)$')
	axis('equal')
	gcf().sans_serif = True

	savefig('radial_gaussianization.tex')
	
	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
