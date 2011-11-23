"""
Visualize random walk in nullspaces.
"""

import sys
import pdb

sys.path.append('./code')

from numpy import *
from models import Distribution, ISA
from tools import Experiment, contours, mapp
from matplotlib.pyplot import *
from numpy.random import seed as np_seed
from random import seed as py_seed

mapp.max_processes = 1

Distribution.VERBOSITY = 0

NUM_VIS = 1
NUM_HID = 3
NUM_SAMPLES = 100000

def main(argv):
	experiment = Experiment(seed=42)

	model = ISA(NUM_VIS, NUM_HID)
	model.initialize()

	Z = model.sample_nullspace(zeros([NUM_VIS, NUM_SAMPLES]) + 3.)

	contours(Z,
		bins=50, 
		threshold=10, 
		levels=[0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56], 
		colors='k')

	random_walk = []

	for k in range(300):
		# reset
		np_seed(42)
		py_seed(42)

		random_walk.append(
			model.sample_nullspace(zeros([NUM_VIS, 1]) + 3, method=('hmc', k)))

	random_walk = hstack(random_walk)

	plot(random_walk[0], random_walk[1], '-r', alpha=0.4)
	plot(random_walk[0], random_walk[1], '.r', markersize=7)
	axis('equal')
	axis([-15, 15, -15, 15])
	xticks([])
	yticks([])

	pdb.set_trace()

	return 0

if __name__ == '__main__':
	sys.exit(main(sys.argv))
