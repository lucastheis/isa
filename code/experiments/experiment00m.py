"""
Train ICA on mixture of ICA toy example.
"""

import sys

sys.path.append('./code')

from models import ISA, GSM
from numpy import *
from numpy.random import *
from pgf import *
from tools import mapp
from copy import deepcopy

mapp.max_processes = 4

PLOT_RANGE = 6
NUM_SAMPLES = 20000

def main(argv):
	seed(2)

	gsm = GSM(1, 4)
	gsm.scales = array([0.1, 1.0, 1.0, 1.0])
	gsm.normalize()

	isa1 = ISA(2, 2)
	isa1.A = eye(2) * 2.
	isa1.subspaces = [gsm, gsm]

	isa2 = ISA(2, 2)
	isa2.A = array([[cos(pi/4.), -sin(pi/4.)], [sin(pi/4.), cos(pi/4.)]])
	isa2.subspaces = [gsm, gsm]

	pr1 = 0.5

	data = hstack([
		isa1.sample(NUM_SAMPLES * pr1),
		isa2.sample(NUM_SAMPLES * (1. - pr1))])
	data = data[:, permutation(data.shape[1])]

	figure()
	plot(data[0, :10000], data[1, :10000], 'k.', opacity=0.1, marker_size=0.5)
	title('true model')
	xlabel('$x_1$')
	ylabel('$x_2$')
	axis([-PLOT_RANGE, PLOT_RANGE, -PLOT_RANGE, PLOT_RANGE])
	draw()



	def callback(model, epoch):
		figure()
		arrow(0, 0, isa1.A[0, 0], isa1.A[1, 0], 'k')
		arrow(0, 0, isa1.A[0, 1], isa1.A[1, 1], 'k')
		arrow(0, 0, isa2.A[0, 0], isa2.A[1, 0], 'k')
		arrow(0, 0, isa2.A[0, 1], isa2.A[1, 1], 'k')
		if isa.noise:
			arrow(0, 0, isa.A[0, 0], isa.A[1, 0], 'b')
			arrow(0, 0, isa.A[0, 1], isa.A[1, 1], 'b')
			arrow(0, 0, isa.A[0, 2], isa.A[1, 2], 'r')
			arrow(0, 0, isa.A[0, 3], isa.A[1, 3], 'r')
			arrow(0, 0, isa.A[0, 4], isa.A[1, 4], 'r')
			arrow(0, 0, isa.A[0, 5], isa.A[1, 5], 'r')
		else:
			arrow(0, 0, isa.A[0, 0], isa.A[1, 0], 'r')
			arrow(0, 0, isa.A[0, 1], isa.A[1, 1], 'r')
			arrow(0, 0, isa.A[0, 2], isa.A[1, 2], 'r')
			arrow(0, 0, isa.A[0, 3], isa.A[1, 3], 'r')
		axis([-PLOT_RANGE, PLOT_RANGE, -PLOT_RANGE, PLOT_RANGE])
		title('bases')
		xlabel('$x_1$')
		ylabel('$x_2$')
		savefig('/Users/lucas/Desktop/bases.{0:0>2}.pdf'.format(epoch))


	isa = ISA(2, 4, noise=False)
	isa.initialize(data)
	isa.initialize(method='laplace')
	isa.train(data, 
		method='lbfgs',
		max_iter=50,
		persistent=True,
		sampling_method=('gibbs', {'num_steps': 2}),
		callback=callback)
#	isa.train_of(data, 
#		max_iter=20,
#		noise_var=0.1,
#		var_goal=1.,
#		beta=10.,
#		step_width=0.01,
#		sigma=1.0,
#		callback=callback)

	# VISUALIZE
	samples = isa.sample(10000)

	figure()
	plot(samples[0], samples[1], 'r.', opacity=0.1, marker_size=0.5)
	title('recovered model')
	xlabel('$x_1$')
	ylabel('$x_2$')
	axis([-PLOT_RANGE, PLOT_RANGE, -PLOT_RANGE, PLOT_RANGE])
	draw()

	figure()
	arrow(0, 0, isa1.A[0, 0], isa1.A[1, 0], 'k')
	arrow(0, 0, isa1.A[0, 1], isa1.A[1, 1], 'k')
	arrow(0, 0, isa1.A[0, 2], isa1.A[1, 2], 'k')
	arrow(0, 0, isa1.A[0, 3], isa1.A[1, 3], 'k')
	arrow(0, 0, isa.A[0, 0], isa.A[1, 0], 'r')
	arrow(0, 0, isa.A[0, 1], isa.A[1, 1], 'r')
	arrow(0, 0, isa.A[0, 2], isa.A[1, 2], 'r')
	arrow(0, 0, isa.A[0, 3], isa.A[1, 3], 'r')
	axis([-PLOT_RANGE, PLOT_RANGE, -PLOT_RANGE, PLOT_RANGE])
	title('bases')
	xlabel('$x_1$')
	ylabel('$x_2$')
	draw()

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
