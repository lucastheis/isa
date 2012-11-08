import sys

sys.path.append('./code')

from tools import preprocess, Experiment, imsave, imformat, stitch
from numpy.fft import *
from numpy import *

def main(argv):
	results = Experiment('results/c_hisa/c_hisa.0.1.xpck')

	model = results['model']
	isa = model.model[1].model
	A = isa.A
	patch_size = int(sqrt(model.dim) + 0.5)

	# preprocessing transforms
	dct = model.transforms[0]
	wt = model.model[1].transforms[0]

	off = 0

	A_fft = []

	for gsm in isa.subspaces():
		A_fft_ = []
		for i in range(off, off + gsm.dim):
			# Fourier transform basis
			A_fft_.append(fftshift(abs(fft2(dot(dct.A[1:].T, A[:, i]).reshape(patch_size, patch_size)))).reshape(-1, 1))

		A_fft.append(mean(A_fft_, 0))
		off += gsm.dim

	A_fft = hstack(A_fft)

	imsave('fft.png',
		stitch(imformat(A_fft.T.reshape(-1, patch_size, patch_size), perc=99, symmetric=False)))

	return 0


if __name__ == '__main__':
	sys.exit(main(sys.argv))
