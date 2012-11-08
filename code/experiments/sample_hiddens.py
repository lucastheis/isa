import sys

sys.path.append('./code')

from tools import preprocess, Experiment, imsave, imformat, stitch
from numpy import zeros, hstack, sqrt, mean, sum, square, dot
from models import StackedModel

def main(argv):
	num_samples = 10

	if len(argv) > 1:
		results = Experiment(argv[1])
	else:
		results = Experiment('results/c_hisa/c_hisa.16x16.1.xpck')

	model = results['model']

	# preprocessing transforms
	wt = model.model[1].transforms[-1]
	dct = model.transforms[-1]

	isa = model.model[1].model

	while isinstance(isa, StackedModel):
		isa = isa.model

	A = isa.A
	off = 0
	samples = []

	for gsm in isa.subspaces():
		print gsm.dim
		B = zeros(A.shape)
		B[:, off:off + gsm.dim] = A[:, off:off + gsm.dim]
		isa.A = B
		samples.append(model.sample(num_samples))
		off += gsm.dim

	samples = hstack(samples)
	# remove DC component and whiten
	samples = dot(dct.A[1:].T, wt(dot(dct.A, samples)[1:]))

	patch_size = int(sqrt(model.dim) + 0.5)
	imsave('sample_hiddens.png',
		stitch(imformat(samples.T.reshape(-1, patch_size, patch_size), perc=99), num_cols=num_samples))

	samples = samples / (0.01 + sqrt(sum(square(samples), 0)))

	imsave('sample_hiddens_normalized.png',
		stitch(imformat(samples.T.reshape(-1, patch_size, patch_size), perc=99), num_cols=num_samples))

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
