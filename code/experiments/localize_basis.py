import sys

sys.path.append('./code')

from tools.isavis import isavis, localize
from tools import preprocess, Experiment, imsave, imformat, stitch
from numpy import dot, hstack, sqrt, load
from numpy.linalg import inv
from isa import ISA

def main(argv):
	results = Experiment('results/c_hisa/c_hisa.0.1.xpck')

	model = results['model']
	isa = model.model[1].model
#	A = isa.A
	dct = model.transforms[0]
	wt = model.model[1].transforms[0]

	# basis in whitened pixel space
#	A = dot(dct.A[1:].T, A)

	# try to find a more localized basis
	B = []
	off = 0
	dims = []

	patch_size = int(sqrt(model.dim) + 0.5)
	data_train = load('data/vanhateren.{0}x{0}.0.npz'.format(patch_size))['data']
	data_train = preprocess(data_train, shuffle=False)[:, :100000]

	states = dot(inv(isa.A), wt(dct(data_train)[1:]))
	
	for gsm in isa.subspaces():
#		B.append(localize(A[:, off:off + gsm.dim]))
		
		ica = ISA(gsm.dim, ssize=1)
		ica.initialize(states[off:off + gsm.dim])
		ica.train(states[off:off + gsm.dim], parameters={
			'verbosity': 1,
			'orthogonalize': True,
			'training_method': 'lbfgs',
			'lbfgs': {
				'max_iter': 50},
			'max_iter': 20})

		B.append(dot(isa.A[:, off:off + gsm.dim], ica.A))

		off += gsm.dim
		dims.append(gsm.dim)

	B = hstack(B)

	# replace basis and make sure likelihood didn't change
	print '{0} [bit/px]'.format(-model.evaluate(data_train))
	isa.A = B
	print '{0} [bit/px]'.format(-model.evaluate(data_train))

	# visualize basis
	B = dot(dct.A[1:].T, B)
	B, num_cols = isavis(B, dims)

	imsave('localized_basis.png',
		stitch(imformat(B.T.reshape(-1, patch_size, patch_size), perc=99), num_cols=num_cols))

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
