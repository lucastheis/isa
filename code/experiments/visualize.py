"""
Visualizes parameters of a learned model.
"""

import sys

sys.path.append('./code')

from tools import Experiment, stitch, imformat
from pgf import *
from numpy import *
from numpy import min, max
from numpy.random import randn

RES = 2
PERC = 99.5

def main(argv):
	experiment = Experiment(argv[1])

	isa = experiment['model'].model[1].model
	dct = experiment['model'].transforms[0]

	# basis in pixel space
	A = dot(dct.A[1:].T, isa.A)

	# sort by norm
	A = A[:, argsort(sqrt(sum(square(A), 0)))[::-1]]

	# adjust intensity range
	a = percentile(abs(A).ravel(), PERC)
	A = (A + a) / (2. * a) * 255. + 0.5
	A[A < 0.] = 0.5
	A[A > 256.] = 255.5
	A = asarray(A, 'uint8')

	# stitch together into a single image
	patch_size = int(sqrt(A.shape[0]) + 0.5)
	patches = stitch(A.T.reshape(-1, patch_size, patch_size))
	patches = repeat(repeat(patches, RES, 0), RES, 1)

	# visualize patches
	figure()
	imshow(patches, dpi=75 * RES)
	title('Basis (white)')
	axis('off')
	draw()



	samples = experiment['model'].sample(128)

	a = percentile(abs(samples).ravel(), PERC)
	samples = (samples + a) / (2. * a) * 255. + 0.5
	samples[samples < 0.] = 0.5
	samples[samples > 256.] = 255.5
	samples = asarray(samples, 'uint8')

	samples = stitch(samples.T.reshape(-1, patch_size, patch_size))
	samples = repeat(repeat(samples, RES, 0), RES, 1)

	# visualize samples
	figure()
	imshow(samples, dpi=75 * RES)
	title('Samples')
	axis('off')
	draw()
	return



	figure()
	samples = []
	for gsm in isa.subspaces:
		samples.append(gsm.sample(1000))

	perc = percentile(hstack(samples), 99.5)
	xvals = linspace(-perc, perc, 200)

	for i in range(8):
		for j in range(16):
			try:
				gsm = isa.subspaces[i * 16 + j]
			except:
				pass
			else:
				subplot(7 - i, j, spacing=0)
				plot(xvals, gsm.loglikelihood(xvals.reshape(1, -1)).ravel(), 'b-', line_width=1.)
				gca().width = 1
				gca().height = 1
				axis([-perc, perc, -5., 2.])
				xtick([])
				ytick([])
	draw()

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
