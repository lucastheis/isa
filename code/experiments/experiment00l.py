"""
Extract patches from images used by Olhausen & Field.
"""

import sys

sys.path.append('./code')

from tools import patchutil
from scipy.io.matlab import loadmat
from numpy import prod, hstack, savez

def main(argv):
	patch_size = int(argv[1]) if len(argv) > 1 else 8
	patch_size = (patch_size, patch_size)

	images = loadmat('data/IMAGES.mat')['IMAGES']

	samples = []

	for i in range(images.shape[2]):
		patches = patchutil.sample(images[:, :, i], patch_size, 10000)
		patches = patches.reshape(prod(patch_size), -1)
		samples.append(patches)

	samples = hstack(samples)
	savez('data/of.{0}x{1}.npz'.format(*patch_size))

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
