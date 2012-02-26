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

	data = []

	for i in range(images.shape[2]):
		samples = patchutil.sample(images[:, :, i], patch_size, 10000)
		samples = samples.reshape(-1, prod(patch_size)).T
		data.append(samples)

	data = hstack(data)
	savez('data/of.{0}x{1}.npz'.format(*patch_size), data=data)

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
