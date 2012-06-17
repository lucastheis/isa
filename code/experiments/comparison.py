"""
Load results for ICA, PoE and other models and generate bar plot.
"""

import sys

sys.path.append('./code')

from tools import Experiment, logmeanexp
from glob import glob
from numpy import *
from pgf import *
from pgf.axes import Axes
from scipy.io.matlab import loadmat

BAR_WIDTH = 0.5

gaussian = {
	'label': '-', 
	'path': 'results/vanhateren/gsm.0.20042012.112721.xpck',
	'color': RGB(0., 0., 0.),
	'fill': RGB(1., 1., 1.),
}

gsm = {
	'label': '-', 
	'path': 'results/vanhateren/gsm.6.20042012.113532.xpck',
	'color': RGB(0., 0., 0.),
	'fill': RGB(1., 1., 1.),
	'pattern': 'crosshatch dots',
}

poe = [
	{
		'label': '2x',
		'path': 'results/vanhateren/poe/AIS_GibbsTrain_white_studentt_L=064_M=128_B=0100000_learner=PMPFdH1_20120523T112449.mat',
		'color': RGB(0., 0., 0.),
		'fill': RGB(0.8, 0.0, 0.0),
	},
	{
		'label': '3x',
		'path': 'results/vanhateren/poe/AIS_GibbsTrain_white_studentt_L=064_M=192_B=0100000_learner=PMPFdH1_20120523T112533.mat',
		'color': RGB(0., 0., 0.),
		'fill': RGB(0.8, 0.0, 0.0),
	},
	{
		'label': '4x',
		'path': 'results/vanhateren/poe/AIS_GibbsTrain_white_studentt_L=064_M=256_B=0100000_learner=PMPFdH1_20120523T112539.mat',
		'color': RGB(0., 0., 0.),
		'fill': RGB(0.8, 0.0, 0.0),
	},
]

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
	{
		'label': '4x',
#		'path': 'results/vanhateren/vanhateren.9.14042012.043802.xpck',
		'path': 'results/vanhateren/vanhateren.9.14042012.copy.xpck',
#		'path': 'results/vanhateren.9/results.1.95.1.xpck',
		'color': RGB(0., 0.0, 0.0),
		'fill': RGB(0., 0.5, 0.8),
	},
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
	'fill': RGB(0.8, 0.0, 0.0),
}

linear_models16 = [
	{
		'label': '1x', 
		'path': 'results/vanhateren/vanhateren.1.25052012.183856.xpck',
		'color': RGB(0., 0., 0.),
		'fill': RGB(0., 0., 0.),
	},
	{
		'label': '2x', 
		'path': 'results/vanhateren.10/results.1.20.xpck',
		'color': RGB(0., 0.0, 0.0),
		'fill': RGB(0., 0.5, 0.8),
	},
]

poe16 = [
	{
		'label': '2x',
		'path': 'results/vanhateren/poe/AIS_GibbsTrain_white_studentt_L=256_M=512_B=0100000_learner=PMPFdH1_20120523T112603.mat',
		'color': RGB(0., 0., 0.),
		'fill': RGB(0.8, 0.0, 0.0),
	},
	{
		'label': '3x',
		'path': 'results/vanhateren/poe/AIS_GibbsTrain_white_studentt_L=256_M=768_B=0100000_learner=PMPFdH1_20120523T112622.mat',
		'color': RGB(0., 0., 0.),
		'fill': RGB(0.8, 0.0, 0.0),
	},
	{
		'label': '4x',
		'path': 'results/vanhateren/poe/AIS_GibbsTrain_white_studentt_L=256_M=1024_B=0100000_learner=PMPFdH1_20120523T112633.mat',
		'color': RGB(0., 0., 0.),
		'fill': RGB(0.8, 0.0, 0.0),
	},
]

def main(argv):
	### 8x8 PATCHES

	subplot(0, 0)

	# LINEAR MODELS

	# load importance weights for each model
	for model in linear_models:
		model['indices'] = []
		model['loglik'] = []

		for path in glob(model['path'][:-4] + '[0-9]*[0-9].xpck'):
			results = Experiment(path)

			if results['ais_weights'].shape[0] not in [1, 200, 300]:
				print path, '(IGNORE)'
				continue

			model['indices'].append(results['indices'])
			model['loglik'].append(logmeanexp(results['ais_weights'], 0).flatten() / log(2.) / 64.)

		# make sure each data point is used only once
		model['indices'] = hstack(model['indices']).tolist()
		model['indices'], idx = unique(model['indices'], return_index=True)
		model['loglik'] = hstack(model['loglik'])[idx]

	# find intersection of data points
	indices = [model['indices'] for model in linear_models]
	indices = list(set(indices[0]).intersection(*indices[1:]))

	print 'Using {0} data points for 8x8 patches.'.format(len(indices))

	# use importance weights to estimate log-likelihood
	for idx, model in enumerate(linear_models):
		subset = [i in indices for i in model['indices']]

		# one estimate of the log-likelihood for each data point
		estimates = model['loglik'][asarray(subset)]

		model['loglik_mean'] = mean(estimates)
		model['loglik_sem'] = std(estimates, ddof=1) / sqrt(estimates.size)

		bar(idx + 2, model['loglik_mean'], 
			yerr=model['loglik_sem'],
			color=model['color'], 
			fill=model['fill'],
			bar_width=BAR_WIDTH,
			pgf_options=[
				'forget plot',
				'nodes near coords', 
				'every node near coord/.style={yshift=0.05cm,font=\\footnotesize}'])



	# PRODUCT OF EXPERTS

	for idx, model in enumerate(poe):
		results = loadmat(model['path'])

		estimates = -results['E'] - results['logZ']
		estimates = estimates.flatten() / 64. / log(2.)
		estimates = estimates[indices]

		model['loglik_mean'] = mean(estimates)
		model['loglik_sem'] = std(estimates, ddof=1) / sqrt(estimates.size)

		bar(idx + 6, model['loglik_mean'], 
			yerr=model['loglik_sem'],
			color=model['color'], 
			fill=model['fill'],
			bar_width=BAR_WIDTH,
			pgf_options=[
				'forget plot',
				'nodes near coords', 
				'every node near coord/.style={yshift=0.05cm,font=\\footnotesize}'])



	# GAUSSIAN SCALE MIXTURE

	results = Experiment(gsm['path'])
	gsm['loglik_mean'] = mean(results['logliks'][:, indices])
	gsm['loglik_sem'] = std(results['logliks'][:, indices], ddof=1) / sqrt(len(indices))

	bar(1, gsm['loglik_mean'], yerr=gsm['loglik_sem'],
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



	xtick(range(len(linear_models) + len(poe) + 2), 
		[gaussian['label']] + \
		[gsm['label']] + \
		[model['label'] for model in linear_models] + \
		[model['label'] for model in poe])
	ytick([0.9, 1.1, 1.3, 1.5])
	xlabel(r'\small Overcompleteness')
	ylabel(r'\small Log-likelihood $\pm$ SEM [bit/pixel]')
	axis(
		width=6,
		height=4,
		ytick_align='outside',
		axis_x_line='bottom',
		axis_y_line='left',
		pgf_options=[
			'xtick style={color=white}',
			r'tick label style={font=\footnotesize}',
			'every outer x axis line/.append style={-}'])
			
	
	axis([-0.5, 8.5, 0.85, 1.65])
	title(r'\small 8 $\times$ 8 image patches')



	### 16x16 PATCHES

	subplot(0, 1)

	# dummy plots
	bar(-1, 0, color=gaussian['color'], fill=gaussian['fill'], bar_width=BAR_WIDTH)
	bar(-1, 0, color=gsm['color'], fill=gsm['fill'], pattern=gsm['pattern'])



	# LINEAR MODELS

	# load importance weights for each model
	for model in linear_models16:
		model['indices'] = []
		model['loglik'] = []

		for path in glob(model['path'][:-4] + '[0-9]*[0-9].xpck'):
			results = Experiment(path)

			model['indices'].append(results['indices'])
			print path
			print mean(logmeanexp(results['ais_weights'], 0).flatten()) / 256. / log(2.)
			print
			model['loglik'].append(logmeanexp(results['ais_weights'], 0).flatten() / 256. / log(2.))

		# make sure each data point is used only once
		model['indices'] = hstack(model['indices']).tolist()
		model['indices'], idx = unique(model['indices'], return_index=True)
		model['loglik'] = hstack(model['loglik'])[idx]

	# find intersection of data points
	indices = [model['indices'] for model in linear_models16]
	indices = list(set(indices[0]).intersection(*indices[1:]))

	print 'Using {0} data points for 16x16 patches.'.format(len(indices))

	# use importance weights to estimate log-likelihood
	for idx, model in enumerate(linear_models16):
		subset = [i in indices for i in model['indices']]

		# exp(ais_weights) represent unbiased estimates of the likelihood
		estimates = model['loglik'][:, asarray(subset)]

		model['loglik_mean'] = mean(estimates)
		model['loglik_sem'] = std(estimates, ddof=1) / sqrt(estimates.size)

		bar(idx + 2, model['loglik_mean'], 
			yerr=model['loglik_sem'],
			color=model['color'], 
			fill=model['fill'],
			bar_width=BAR_WIDTH,
			pgf_options=[
				'forget plot',
				'nodes near coords', 
				'every node near coord/.style={yshift=0.05cm,font=\\footnotesize}'])



	# PRODUCT OF EXPERTS

	for idx, model in enumerate(poe16):
		results = loadmat(model['path'])

		estimates = -results['E'] - results['logZ']
		estimates = estimates.flatten() / 256. / log(2.)
		estimates = estimates[indices]

		model['loglik_mean'] = mean(estimates)
		model['loglik_sem'] = std(estimates, ddof=1) / sqrt(estimates.size)

		bar(idx + 4, model['loglik_mean'], 
			yerr=model['loglik_sem'],
			color=model['color'], 
			fill=model['fill'],
			bar_width=BAR_WIDTH,
			pgf_options=[
				'forget plot',
				'nodes near coords', 
				'every node near coord/.style={yshift=0.05cm,font=\\footnotesize}'])



	# GAUSSIAN SCALE MIXTURE

	results = Experiment(gsm16['path'])
	gsm['loglik_mean'] = mean(results['logliks'][:, indices])
	gsm['loglik_sem'] = std(results['logliks'][:, indices], ddof=1) / sqrt(len(indices))

	bar(1, gsm['loglik_mean'], yerr=gsm['loglik_sem'],
		color=gsm16['color'],
		fill=gsm16['fill'],
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



	xtick([0, 1, 2, 3, 4, 5, 6], ['-', '-', '1x', '2x', '2x', '3x', '4x'])
	ytick([0.9, 1.1, 1.3, 1.5])
	xlabel(r'\small Overcompleteness')
	axis(
		width=4.8,
		height=4,
		ytick_align='outside',
		axis_x_line='bottom',
		axis_y_line='left',
		pgf_options=[
			'xtick style={color=white}',
			r'tick label style={font=\footnotesize}',
			'every outer x axis line/.append style={-}'])
	axis([-0.5, 6.5, 0.85, 1.65])
	title(r'\small 16 $\times$ 16 image patches')

	gcf().margin = 4
	gcf().save('results/vanhateren/comparison.tex')

	# dummy plots
	bar(-1, 0, color=linear_models[0]['color'], fill=linear_models[0]['fill'])
	bar(-1, 0, color=linear_models[1]['color'], fill=linear_models[1]['fill'])
	bar(-1, 0, color=poe[0]['color'], fill=poe[0]['fill'])

	legend('Gaussian', 'GSM', 'LM', 'OLM', 'PoT', location='outer north east')

	savefig('results/vanhateren/comparison.tex')
	draw()

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
