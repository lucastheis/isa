"""
Visualize overcompleteness of different models.
"""

import sys

sys.path.append('./code')

from tools import Experiment, stitch, imformat
from scipy.stats import laplace
from pgf import *
from numpy import *

models = [
	{
		'path': 'results/vanhateren/vanhateren.5.08042012.041413.xpck',
		'legend': 'Laplace, 2x',
		'color': RGB(0.8, 0.5, 0.0),
	},
	{
		'path': 'results/vanhateren/vanhateren.7.08042012.150147.xpck',
		'legend': 'GSM, 2x',
		'color': RGB(0., 0.7, 1.0),
	},
	{
		'path': 'results/vanhateren/vanhateren.8.09042012.063158.xpck',
		'legend': 'GSM, 3x',
		'color': RGB(0., 0.5, 0.8),
	},
	{
#		'path': 'results/vanhateren/vanhateren.9.14042012.043802.xpck',
		'path': 'results/vanhateren.9/results.1.10.2.xpck',
		'legend': 'GSM, 4x',
		'color': RGB(0., 0.2, 0.5),
	},
]

# model used for visualizing basis
model = {
	'path': 'results/vanhateren/vanhateren.9.14042012.043802.xpck',
#	'path': 'results/vanhateren.9/results.1.10.2.xpck',
}

# resolution of basis image
RES = 8

# controls range of displayed gray levels
PERC = 99.5

NUM_COLS = 16
 
def main(argv):
	### PLOT BASIS VECTOR NORMS

	subplot(0, 0, spacing=1.)

	legend_entries = []

	for model in models:
		results = Experiment(model['path'])

		isa = results['model'].model[1].model
		dct = results['model'].transforms[0]

		# basis in whitened pixel space
		A = dot(dct.A[1:].T, isa.A)

		# basis vector norms
		norms = sort(sqrt(sum(square(A), 0)))[::-1]

		plot(norms,
			color=model['color'],
			line_width=1.2)
		plot([len(norms), len(norms) + 1, 255], [norms[-1], 0, 0],
			color=model['color'],
			line_style='densely dashed',
			line_width=1.2,
			pgf_options=['forget plot'])

		legend_entries.append(model['legend'])

	xlabel('Basis coefficient, $i$')
	ylabel('Basis vector norm, $||a_i||$')
	legend(*legend_entries, location='north east')
	axis(width=5, height=4)
	axis([0, 256, 0, 1])
	xtick([64, 128, 192, 256])
	grid()



	### VISUALIZE BASIS

	subplot(0, 1)

	results = Experiment(model['path'])

	isa = results['model'].model[1].model
	dct = results['model'].transforms[0]

	# basis in whitened pixel space
	A = dot(dct.A[1:].T, isa.A)
	indices = argsort(sqrt(sum(square(A), 0)))[::-1]
	A = A[:, indices]

	# adjust intensity range
	a = percentile(abs(A).ravel(), PERC)
	A = (A + a) / (2. * a) * 255. + 0.5
	A[A < 0.] = 0.5
	A[A > 256.] = 255.5
	A = asarray(A, 'uint8')

	# stitch together into a single image
	patch_size = int(sqrt(A.shape[0]) + 0.5)
	patches = stitch(A.T.reshape(-1, patch_size, patch_size), num_cols=NUM_COLS)
	patches = repeat(repeat(patches, RES, 0), RES, 1)

	imshow(patches, dpi=75 * RES)
	rectangle(72 * RES, 90 * RES, 64 * RES, 46 * RES,
		color=RGB(1.0, 0.8, 0.5),
		line_width=1.,
		line_style='densely dashed')

	axis(
		height=4, 
		width=4,
		ticks='none', 
		axis_on_top=False,
		clip=False,
		pgf_options=['xlabel style={yshift=-0.47cm}', 'clip=false'])

	savefig('results/vanhateren/overcompleteness.tex')
	draw()



	### MARGINAL SOURCE DISTRIBUTIONS

	figure()
	samples = []
	for gsm in isa.subspaces:
		samples.append(gsm.sample(1000))

	perc = percentile(hstack(samples), 99.5)
	xvals = linspace(-perc, perc, 100)

	for i in range(1, 6):
		for j in range(8, 15):
			try:
				gsm = isa.subspaces[indices[i * NUM_COLS + j]]
			except:
				pass
			else:
				subplot(7 - i, j, spacing=0)
				plot(xvals, laplace.logpdf(xvals, scale=sqrt(0.5)).ravel(), 'k', opacity=0.5)
				plot(xvals, gsm.loglikelihood(xvals.reshape(1, -1)).ravel(), 'b-', line_width=1.)
				gca().width = 0.8
				gca().height = 0.8
				axis([-perc, perc, -6., 2.])
				xtick([])
				ytick([])

	savefig('results/vanhateren/marginals.tex')
	draw()

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
