"""
Fit Gaussian scale mixture to van Hateren patches.
"""

import sys

sys.path.append('./code')

from tools import Experiment, preprocess
from models import GSM, StackedModel, ConcatModel, MoGaussian
from transforms import LinearTransform, WhiteningTransform
from numpy import load, mean, std, sqrt, log

parameters = [
	# Gaussian
	['8x8',    1,   1, False],
	['16x16',  1,   1, False],
	['8x8',    1, 100, True],
	['16x16',  1, 100, True],

	# GSM
	['8x8',   20, 100, False],
	['16x16', 20, 100, False],
	['8x8',   20, 100, True],
	['16x16', 20, 100, True],
]
 
def main(argv):
	if len(argv) < 2:
		print 'Usage:', argv[0], '<param_id>'
		print
		print '  {0:>3} {1:>7} {2:>5} {3:>5} {4:>5}'.format(
			'ID', 'PS', 'NS', 'TI', 'DC')

		for id, params in enumerate(parameters):
			print '  {0:>3} {1:>7} {2:>5} {3:>5} {4:>5}'.format(id, *params)

		print
		print '  ID = parameter set'
		print '  PS = patch size'
		print '  NS = number of scales'
		print '  TI = number of training iterations'
		print '  DC = model DC component separately'

		return 0

	# start experiment
	experiment = Experiment(server='10.38.138.150')

	# hyperparameters
	patch_size, num_scales, max_iter, separate_dc = parameters[int(argv[1])]



	### DATA PREPROCESSING

	# load data, log-transform and center data
	data = load('data/vanhateren.{0}.1.npz'.format(patch_size))['data']
	data = data[:, :100000]
	data = preprocess(data)



	### MODEL DEFINITION AND TRAINING

	if separate_dc:
		# discrete cosine transform and symmetric whitening transform
		dct = LinearTransform(dim=int(sqrt(data.shape[0])), basis='DCT')
		wt = WhiteningTransform(dct(data)[1:], symmetric=True)

		model = StackedModel(dct, ConcatModel(
			MoGaussian(20), 
			StackedModel(wt, GSM(data.shape[0] - 1, num_scales))))

	else:
		# symmetric whitening transform
		wt = WhiteningTransform(data, symmetric=True)
		model = StackedModel(wt, GSM(data.shape[0], num_scales))



	### MODEL TRAINING AND EVALUATION

	model.train(data, max_iter=max_iter, tol=1e-7)

	# load and preprocess test data
	data = load('data/vanhateren.{0}.0.npz'.format(patch_size))['data']
	data = preprocess(data, shuffle=False)

	# log-likelihod in [bit/pixel]
	logliks = model.loglikelihood(data) / log(2.) / data.shape[0]
	loglik = mean(logliks)
	sem = std(logliks, ddof=1) / sqrt(logliks.shape[1])

	print 'log-likelihood: {0:.4f} +- {1:.4f} [bit/pixel]'.format(loglik, sem)

	experiment['logliks'] = logliks
	experiment['loglik'] = loglik
	experiment['sem'] = sem
	experiment.save('results/vanhateren/gsm.{0}.{{0}}.{{1}}.xpck'.format(argv[1]))

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
