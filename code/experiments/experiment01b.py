"""
Finds experiments with certain properties.
"""

import os
import sys

sys.path.append('./code')

from getopt import getopt
from tools import Experiment
from numpy import *

filepath = './results/experiment01a/'

def main(argv):
	try:
		opts, _ = getopt(argv[1:], 's:o:p:h', ['help'])
		opts = dict(opts)
	except getopt.GetoptError:
		opts = {'-h': ''}

	if '--help' in opts or '-h' in opts:
		print 'Usage:', argv[0], '[-h] [-s subspace_size] [-o overcompleteness] [-p patch_size]'
		return 0

	for filename in os.listdir(filepath):
		if filename.endswith('.xpck'):
			results = Experiment(os.path.join(filepath, filename))

			if '-o' in opts:
				if results['model'].num_hiddens / results['model'].num_visibles != int(opts['-o']):
					continue
			if '-s' in opts:
				if any([gsm.dim != int(opts['-s']) for gsm in results['model'].subspaces]):
					continue
			if '-p' in opts:
				if opts['-p'] not in results['parameters']:
					continue

			print filename

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
