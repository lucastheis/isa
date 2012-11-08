"""
Save preprocessed training and test sets as *.mat files.
"""

import sys

sys.path.append('./code')

from tools import preprocess, Experiment, mapp, imsave, imformat, stitch
from scipy.io.matlab import savemat
from numpy import load

def main(argv):
	for patch_size in ['8x8', '16x16']:
		data = load('data/vanhateren.{0}.0.npz'.format(patch_size))['data']
		data = preprocess(data)

		savemat('data/vanhateren.{0}.test.mat'.format(patch_size), {'data': data})

		data = load('data/vanhateren.{0}.1.npz'.format(patch_size))['data']
		data = preprocess(data)

		savemat('data/vanhateren.{0}.train.mat'.format(patch_size), {'data': data})

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
