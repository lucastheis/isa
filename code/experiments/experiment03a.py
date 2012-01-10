"""
Train and evaluate Gaussian scale mixture.
"""

import sys

sys.path.append('./code')

from models import ConcatModel, StackedModel, GSM, MoGaussian, Distribution
from transforms import LinearTransform, WhiteningTransform
from tools import preprocess, Experiment
from numpy import load, sqrt

parameters = [
	['8x8',   10, 10, 32],
	['16x16', 10, 10, 32]
	]

Distribution.VERBOSITY = 3

def main(argv):
	if len(argv) < 2:
		print 'Usage:', argv[0], '<param_id>'
		return 0

	experiment = Experiment()

	# hyperparameters
	patch_size, num_components, num_scales, noise_level = parameters[int(argv[1])]



	### TRAINING PHASE

	# load data, log-transform and center data
	data = load('data/vanhateren.{0}.1.npz'.format(patch_size))['data']
	data = preprocess(data, noise_level=noise_level)
	
	# apply DCT to data
	dct = LinearTransform(dim=int(sqrt(data.shape[0])), basis='DCT')
	data = dct(data)

	# whitening transform
	wt = WhiteningTransform(data[1:], symmetric=True)

	# model DC component separately
	model = ConcatModel(
		MoGaussian(num_components), 
		StackedModel(wt, GSM(data.shape[0] - 1, num_scales)))

	# train mixture distribution on DC component
	model.train(data, 0, max_iter=100)
	
	# train Gaussian scale mixture
	model.train(data, 1, max_iter=100)



	### TESTING PHASE

	data = load('data/vanhateren.{0}.0.npz'.format(patch_size))['data']
	data = preprocess(data, noise_level=noise_level)

	data = dct(data)

	logloss = model.evaluate(data)
	print '{0:.4f} [bit/pixel]'.format(logloss)

	logloss = model[1].model.evaluate(wt(data[1:]))
	print '{0:.4f} [bit/pixel] (white)'.format(logloss)

	experiment['parameters'] = parameters[int(argv[1])]
	experiment['transforms'] = [dct, wt]
	experiment['model'] = model
	experiment['logloss'] = logloss
	experiment.save('results/experiment03a/experiment03a.{0}.{1}.xpck')

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
