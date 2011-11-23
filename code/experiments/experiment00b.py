"""
Plot likelihood functions of multivariate Student parameter.
"""

import sys

sys.path.append('./code')

from tools.parallel import mapp
from models import ISA
from numpy import zeros
from cProfile import run

mapp.max_processes = 1

NUM_VIS = 64
NUM_HID = 128
NUM_SAMPLES = 1000

def main(argv):
	run('ISA(NUM_VIS, NUM_HID).sample_posterior(zeros([NUM_VIS, NUM_SAMPLES]))',
		sort='cumulative')

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
