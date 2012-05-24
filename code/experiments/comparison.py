"""
Compare overcomplete ICA and PoE and plot results.
"""

import sys

sys.path.append('./code')

from tools import Experiment, logmeanexp
from glob import glob
from numpy import *
from pgf import *
from pgf.axes import Axes

BAR_WIDTH = 0.5

gaussian = {
	'label': '', 
	'path': 'results/vanhateren/gsm.0.20042012.112721.xpck',
	'color': RGB(0., 0., 0.),
	'fill': RGB(1., 1., 1.),
}

gsm = {
	'label': '', 
	'path': 'results/vanhateren/gsm.6.20042012.113532.xpck',
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
#	{
#		'label': '4x',
#		'path': 'results/vanhateren/vanhateren.9.14042012.043802.xpck',
#		'color': RGB(0., 0.0, 0.0),
#		'fill': RGB(0., 0.5, 0.8),
#	},
]

gaussian16 = {
	'label': '1x', 
	'path': 'results/vanhateren/gsm.1.24052012.184339.xpck',
	'color': RGB(0., 0., 0.),
	'fill': RGB(1., 1., 1.),
}

gsm16 = {
	'label': '1x', 
	'path': 'results/vanhateren/gsm.7.24052012.184817.xpck',
	'color': RGB(0., 0., 0.),
	'fill': RGB(1., 1., 1.),
	'pattern': 'crosshatch dots',
}

poe16 = {
	'label': '4x',
	'path': 'results/vanhateren/poe.xpck',
	'color': RGB(0., 0., 0.),
	'fill': RGB(0.8, 0.9, 0.0),
}

linear_models16 = [
#	{
#		'label': '1x', 
#		'path': 'results/vanhateren/vanhateren.0.13042012.024853.xpck',
#		'color': RGB(0., 0., 0.),
#		'fill': RGB(0., 0., 0.),
#	},
#	{
#		'label': '2x', 
#		'path': 'results/vanhateren/vanhateren.7.08042012.150147.xpck',
#		'color': RGB(0., 0.0, 0.0),
#		'fill': RGB(0., 0.5, 0.8),
#	},
]

def main(argv):
	### 8x8 PATCHES

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
	indices = list(set(indices[0]).intersection(*indices[1:]))

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
			pgf_options=[
				'forget plot',
				'nodes near coords', 
				'every node near coord/.style={yshift=0.05cm,font=\\footnotesize}'])

	bar(4, 0.9,
		labels='?',
		color=linear_models[1]['color'],
		fill=linear_models[1]['fill'],
		bar_width=BAR_WIDTH,
		pgf_options=['forget plot'])



	# PRODUCT OF EXPERTS

	bar(5, 0.9,
		labels='?',
		color=poe['color'],
		fill=poe['fill'],
		bar_width=BAR_WIDTH,
		pgf_options=['forget plot'])



	# GAUSSIAN SCALE MIXTURE

	results = Experiment(gsm['path'])
	gsm['loglik_mean'] = mean(results['logliks'][:, indices])
	gsm['loglik_sem'] = std(results['logliks'][:, indices], ddof=1) / sqrt(len(indices))

	bar(6, gsm['loglik_mean'], yerr=gsm['loglik_sem'],
		color=gsm['color'],
		fill=gsm['fill'],
		bar_width=BAR_WIDTH,
		pattern=gsm['pattern'],
		pgf_options=[
			'forget plot',
			'nodes near coords',
			'every node near coord/.style={yshift=0.05cm, font=\\footnotesize}'])
	


	# GAUSSIAN

	results = Experiment(gaussian['path'])
	gaussian['loglik_mean'] = mean(results['logliks'][:, indices])
	gaussian['loglik_sem'] = std(results['logliks'][:, indices], ddof=1) / sqrt(len(indices))

	bar(0, gaussian['loglik_mean'], yerr=gaussian['loglik_sem'],
		color=gaussian['color'],
		fill=gaussian['fill'],
		bar_width=BAR_WIDTH,
		pgf_options=['nodes near coords', 'every node near coord/.style={yshift=0.05cm, font=\\footnotesize}'])



	xtick(range(len(linear_models) + 4), 
		[gaussian['label']] + \
		[model['label'] for model in linear_models] + ['4x'] + \
		[poe['label']] + \
		[gsm['label']])
	ytick([0.9, 1.1, 1.3, 1.5])
	xlabel(r'\small Overcompleteness')
	ylabel(r'\small Log-likelihood $\pm$ SEM [bit/pixel]')
	axis(
		width=5,
		height=4,
		ytick_align='outside',
		pgf_options=['xtick style={color=white}', r'tick label style={font=\footnotesize}'])
			
	
	axis([-0.5, 6.5, 0.85, 1.55])
	title(r'\small 8 $\times$ 8 image patches')



	### 16x16 PATCHES

	subplot(0, 1)

	# dummy plots
	bar(-1, 0, color=gaussian['color'], fill=gaussian['fill'], bar_width=BAR_WIDTH)
	bar(-1, 0, color=linear_models[0]['color'], fill=linear_models[0]['fill'])

	# LINEAR MODELS

	bar(1, 0.9,
		labels='?',
		color=linear_models[0]['color'],
		fill=linear_models[0]['fill'],
		bar_width=BAR_WIDTH,
		pgf_options=['forget plot'])

	bar(2, 0.9,
		labels='?',
		color=linear_models[1]['color'],
		fill=linear_models[1]['fill'],
		bar_width=BAR_WIDTH,
		pgf_options=['forget plot'])



	# PRODUCT OF EXPERTS

	bar(3, 0.9,
		labels='?',
		color=poe['color'],
		fill=poe['fill'],
		bar_width=BAR_WIDTH,
		pgf_options=['forget plot'])



	# GAUSSIAN SCALE MIXTURE

	results = Experiment(gsm16['path'])
	gsm['loglik_mean'] = mean(results['logliks'][:, indices])
	gsm['loglik_sem'] = std(results['logliks'][:, indices], ddof=1) / sqrt(len(indices))

	bar(4, gsm['loglik_mean'], yerr=gsm['loglik_sem'],
		color=gsm['color'],
		fill=gsm['fill'],
		bar_width=BAR_WIDTH,
		pattern=gsm['pattern'],
		pgf_options=[
			'forget plot',
			'nodes near coords',
			'every node near coord/.style={yshift=0.05cm, font=\\footnotesize}'])



	# GAUSSIAN

	results = Experiment(gaussian16['path'])
	gaussian['loglik_mean'] = mean(results['logliks'][:, indices])
	gaussian['loglik_sem'] = std(results['logliks'][:, indices], ddof=1) / sqrt(len(indices))

	bar(0, gaussian['loglik_mean'], yerr=gaussian['loglik_sem'],
		color=gaussian['color'],
		fill=gaussian['fill'],
		bar_width=BAR_WIDTH,
		pgf_options=['forget plot', 'nodes near coords', 'every node near coord/.style={yshift=0.05cm, font=\\footnotesize}'])



	xtick([0, 1, 2, 3, 4], ['', '1x', '2x', '2x', ''])
	ytick([0.9, 1.1, 1.3, 1.5])
	xlabel(r'\small Overcompleteness')
	ylabel(r'\small Log-likelihood $\pm$ SEM [bit/pixel]')
	axis(
		width=3.6,
		height=4,
		ytick_align='outside',
		pgf_options=['xtick style={color=white}', r'tick label style={font=\footnotesize}'])
	axis([-0.5, 4.5, 0.85, 1.55])
	title(r'\small 16 $\times$ 16 image patches')

	gcf().margin = 4
	gcf().save('results/vanhateren/comparison.tex')

	# dummy plots
	bar(-1, 0, color=linear_models[1]['color'], fill=linear_models[1]['fill'])
	bar(-1, 0, color=poe['color'], fill=poe['fill'])
	bar(-1, 0, color=gsm['color'], fill=gsm['fill'], pattern=gsm['pattern'])

	legend('Gaussian', 'LM', 'OLM', 'PoT', 'GSM', location='outer north east')

	draw()

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
