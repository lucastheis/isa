"""
Compare overcomplete ICA and PoE and plot results.
"""

import sys

sys.path.append('./code')

from tools import Experiment, logmeanexp
from glob import glob
from numpy import *
from pgf import *

BAR_WIDTH = 0.7

gaussian = {
	'label': '1x', 
	'path': 'results/vanhateren/gaussian.xpck',
	'color': RGB(0., 0., 0.),
	'fill': RGB(1., 1., 1.),
}

gsm = {
	'label': '1x', 
	'path': 'results/vanhateren/gsm.xpck',
	'color': RGB(0., 0., 0.),
	'fill': RGB(1., 1., 1.),
	'pattern': 'crosshatch dots',
}

poe = {
	'label': '4x',
	'path': 'results/vanhateren/poe.xpck',
	'color': RGB(0., 0., 0.),
	'fill': RGB(0.8, 0.9, 0.0),
}

linear_models = [
	{
		'label': '1x', 
		'path': 'results/vanhateren/vanhateren.0.13042012.024853.xpck',
		'color': RGB(0., 0., 0.),
		'fill': RGB(0., 0., 0.),
	},
	{
		'label': '2x', 
		'path': 'results/vanhateren/vanhateren.7.08042012.150147.xpck',
		'color': RGB(0., 0.0, 0.0),
		'fill': RGB(0., 0.5, 0.8),
	},
	{
		'label': '3x',
		'path': 'results/vanhateren/vanhateren.8.09042012.063158.xpck',
		'color': RGB(0., 0.0, 0.0),
		'fill': RGB(0., 0.5, 0.8),
	},
]

def main(argv):
	subplot(0, 0)

	# LINEAR MODELS

	# load importance weights for each models model
	for model in linear_models:
		model['indices'] = []
		model['ais_weights'] = []

		for path in glob(model['path'][:-4] + '[0-9]*[0-9].xpck'):
			results = Experiment(path)

			model['indices'].append(results['indices'])
			model['ais_weights'].append(results['ais_weights'])

		# make sure each data point is used only once
		model['indices'] = hstack(model['indices']).tolist()
		model['indices'], idx = unique(model['indices'], return_index=True)
		model['ais_weights'] = hstack(model['ais_weights'])[:, idx]

	# find intersection of data points
	indices = [model['indices'] for model in linear_models]
	indices = set(indices[0]).intersection(*indices[1:])

	# use importance weights to estimate log-likelihood
	for idx, model in enumerate(linear_models):
		subset = [i in indices for i in model['indices']]

		# exp(ais_weights) represent unbiased estimates of the likelihood
		estimates = model['ais_weights'][:, asarray(subset)]

		model['loglik_mean'] = mean(logmeanexp(estimates, 0))
		model['loglik_sem'] = std(logmeanexp(estimates, 0), ddof=1) / sqrt(model['ais_weights'].shape[1])

		bar(idx + 1, model['loglik_mean'], 
			yerr=model['loglik_sem'],
			color=model['color'], 
			fill=model['fill'],
			bar_width=BAR_WIDTH,
			pgf_options=['forget plot', 'nodes near coords'])

	bar(4, 1.0,
		labels='?',
		color=linear_models[1]['color'],
		fill=linear_models[1]['fill'],
		bar_width=BAR_WIDTH,
		pgf_options=['forget plot'])



	# PRODUCT OF EXPERTS

	bar(5, 1.0,
		labels='?',
		color=poe['color'],
		fill=poe['fill'],
		bar_width=BAR_WIDTH,
		pgf_options=['forget plot'])



	# GAUSSIAN SCALE MIXTURE

	bar(6, 1.35,
		color=gsm['color'],
		fill=gsm['fill'],
		bar_width=BAR_WIDTH,
		pattern=gsm['pattern'],
		pgf_options=['forget plot', 'nodes near coords'])
	


	# GAUSSIAN

	bar(0, 1.0,
		color=gaussian['color'],
		fill=gaussian['fill'],
		bar_width=BAR_WIDTH,
		pgf_options=['nodes near coords'])

	xtick(range(len(linear_models) + 4), 
		[gaussian['label']] + \
		[model['label'] for model in linear_models] + ['4x'] + \
		[poe['label']] + \
		[gsm['label']])
	xlabel('Overcompleteness')
	ylabel('Log-likelihood $\pm$ SEM [bit/pixel]')
	axis(width=8, height=6)
	axis([-0.5, 6.5, 0.95, 1.45])
	title('8 $\\times$ 8 image patches')

	subplot(0, 1)

	xtick([0, 1, 2, 3, 4], ['1x', '1x', '2x', '2x', '1x'])
	xlabel('Overcompleteness')
	ylabel('Log-likelihood $\pm$ SEM [bit/pixel]')
	axis(width=5.7, height=6)
	axis([-0.5, 4.5, 0.95, 1.45])
	title('16 $\\times$ 16 image patches')

	# dummy plots
	bar(-1, 0, color=gaussian['color'], fill=gaussian['fill'], bar_width=BAR_WIDTH)
	bar(-1, 0, color=linear_models[0]['color'], fill=linear_models[0]['fill'])
	bar(-1, 0, color=linear_models[1]['color'], fill=linear_models[1]['fill'])
	bar(-1, 0, color=poe['color'], fill=poe['fill'])
	bar(-1, 0, color=gsm['color'], fill=gsm['fill'], pattern=gsm['pattern'])

	legend('Gaussian', 'ICA', 'OICA', 'PoE', 'GSM', location='outer north east')

	gcf().margin = 4
	draw()

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
