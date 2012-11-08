import os
import re
import sys

sys.path.append('./code')

from tools import Experiment
from glob import glob
from pgf import *
from numpy import *

filepath = 'results/c_vanhateren.18/'

def main(argv):
	for f in sorted(glob(filepath + 'results.1.*.xpck'), key=os.path.getmtime):
		results = Experiment(f)
		
		isa = results['model'].model[1].model
		samples = isa.subspaces()[3].sample(100000)

		figure()
		hist(sqrt(sum(square(samples), 0)), 200)
		xlabel('radial component, r')
		ylabel('$p(ru)$')
		savefig(filepath + 'radial_dist/{0}.pdf'.format(
			re.match(filepath + 'results\.1\.(?P<iteration>[0-9]+)\.xpck', f).group(1)))
	
	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
