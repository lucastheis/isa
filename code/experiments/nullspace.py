"""
Visualizes a one-dimensional null space distribution.
This will take a couple of minutes.
"""

import sys

sys.path.append('./code')

from models import ISA
from numpy import *
from numpy import min, max
from numpy.random import *
from numpy.linalg import *
from pgf import *
from tools import mapp

# disable parallelization
mapp.max_processes = 1

# histogram parameters
NUM_SAMPLES = 1000000
NUM_BINS = 800
MCMC_STEPS = 10
Z_FROM = -20.
Z_TO = 30.

# size and resolution for image of prior
IMG_SIZE = 1024
DPI = 300

def main(argv):
	seterr(over='raise', divide='raise', invalid='raise')
	
	# OICA with Student's t-distribution marginals
	ica = ISA(1, 2, ssize=1, num_scales=20)
	ica.A[:] = [0.7, 1.1]

	# fit marginals to exponential power distribution
	ica.initialize(method='exponpow')

	# prior landscape
	xmin, xmax = -35, 35
	s = meshgrid(linspace(xmin, xmax, IMG_SIZE), linspace(xmin, xmax, IMG_SIZE))
	S = vstack([s[0].flatten(), s[1].flatten()])
	E = ica.prior_energy(S).reshape(*s[0].shape)[::-1]

	# nullspace
	W = pinv(ica.A)
	V = pinv(ica.nullspace_basis())
	x = 18.
	s_fr = (W * x + V * Z_FROM).flatten()
	s_to = (W * x + V * Z_TO).flatten()

	# sample nullspace
	Z = ica.sample_nullspace(zeros([1, NUM_SAMPLES]) + x,
		method=('gibbs', {'num_steps': MCMC_STEPS})).flatten()

	figure()
	imshow(-E,
		cmap='shadows', 
		dpi=DPI,
		vmin=-7.0,
		vmax=-2.0,
		limits=[xmin, xmax, xmin, xmax])
	plot([s_fr[0], s_to[0]], [s_fr[1], s_to[1]], line_width=3., color='cyan')
	arrow(0, 0, W[0, 0] * x, W[1, 0] * x, line_width=1.5)
	text(5.3, 5., '$A^+$')
	axis('origin')
	xtick([])
	ytick([])
	xlabel('$s_1$')
	ylabel('$s_2$')
	savefig('results/prior.tex')
	draw()

	figure()
	h = hist(Z, NUM_BINS, density=True, color='cyan', opacity=0.8, line_width=0.)
	h.const_plot = False
	axis('origin')
	axis([Z_FROM, Z_TO, 0., 0.14])
	xlabel('$z$')
	ylabel('$p(z \mid x)$')
	xtick([])
	ytick([])
	savefig('results/nullspace.tex')
	draw()

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
