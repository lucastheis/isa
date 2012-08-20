import sys

sys.path.append('./code')

from tools import Experiment, imsave, imformat, stitch
from numpy import *
from numpy.linalg import inv
from numpy.random import *

num_data = 100

def main(argv):
	results = Experiment('results/c_hmoisa/results.3.170.xpck')

	model = results['model']

	isa = model.model[1].model
	rg = model.model[1].transforms[0]
	wt = model.transforms[0]

	states = isa.hidden_states()

	print states.shape[1], 'hidden states'

	norms = zeros([states.shape[0] / 2, states.shape[1]])
	phases = zeros([states.shape[0] / 2, states.shape[1]])

	# compute norms and phases
	for i in range(states.shape[0] / 2):
		norms[i, :] = sqrt(sum(square(states[2 * i:2 * i + 2, :]), 0))
		phases[i, :] = arctan2(states[2 * i + 1, :], states[2 * i, :])

#	# shuffle norms and phases between data points
#	phases_rnd = phases[:, permutation(phases.shape[1])]
#	norms_rnd = norms[:, permutation(norms.shape[1])]
#
#	states_rnd_phase = zeros(states.shape)
#	states_rnd_norm = zeros(states.shape)
#	states_rnd_sub = zeros(states.shape)
	states_knn_phase = zeros([states.shape[0], num_data])
#	states_knn_norm = zeros([states.shape[0], num_data])

	for j in range(num_data):
		# determine data point with closest norms
		d = sum(abs(norms - norms[:, [j]]), 0)
		d[j] = inf
		k = argmin(d)

		for i in range(states.shape[0] / 2):
			states_knn_phase[2 * i:2 * i + 2, j] = [
				cos(phases[i, k]) * norms[i, j],
				sin(phases[i, k]) * norms[i, j]]
#
#	for j in range(num_data):
#		# determine data point with closest phases
#		d = sum(1. - cos(phases - phases[:, [j]]), 0)
#		d[j] = inf
#		k = argmin(d)
#
#		for i in range(states.shape[0] / 2):
#			states_knn_norm[2 * i:2 * i + 2, j] = [
#				cos(phases[i, j]) * norms[i, k],
#				sin(phases[i, j]) * norms[i, k]]
#
#	# reconstruct states
#	for i in range(states.shape[0] / 2):
#		states_rnd_phase[2 * i:2 * i + 2, :] = [
#			cos(phases_rnd[i, :]) * norms[i, :],
#			sin(phases_rnd[i, :]) * norms[i, :]]
#		states_rnd_norm[2 * i:2 * i + 2, :] = [
#			cos(phases[i, :]) * norms_rnd[i, :],
#			sin(phases[i, :]) * norms_rnd[i, :]]
#		states_rnd_sub[2 * i:2 * i + 2, :] = states[2 * i:2 * i + 2, permutation(states.shape[1])]
#
#	states = states[:, :num_data]
#	states_rnd_phase = states_rnd_phase[:, :num_data]
#	states_rnd_norm = states_rnd_norm[:, :num_data]
#	states_rnd_sub = states_rnd_sub[:, :num_data]

	# map hidden states into pixel space
#	data = dot(inv(wt.A)[:, 1:1 + isa.dim], rg.inverse(dot(isa.A, states)))
#	data_rnd_phase = dot(inv(wt.A)[:, 1:1 + isa.dim], rg.inverse(dot(isa.A, states_rnd_phase)))
#	data_rnd_norm = dot(inv(wt.A)[:, 1:1 + isa.dim], rg.inverse(dot(isa.A, states_rnd_norm)))
#	data_rnd_sub = dot(inv(wt.A)[:, 1:1 + isa.dim], rg.inverse(dot(isa.A, states_rnd_sub)))
	data_knn_phase = dot(inv(wt.A)[:, 1:1 + isa.dim], rg.inverse(dot(isa.A, states_knn_phase)))
#	data_knn_norm = dot(inv(wt.A)[:, 1:1 + isa.dim], rg.inverse(dot(isa.A, states_knn_norm)))
#	filters = dot(inv(wt.A)[:, 1:1 + isa.dim], rg.inverse(isa.A))
#
#	samples = model.sample(100)
#
#	imsave('results/c_hmoisa/filters.png',
#		stitch(imformat(filters.T.reshape(-1, 20, 20), perc=99.5), num_rows=20))
#	imsave('results/c_hmoisa/data.png',
#		stitch(imformat(data.T.reshape(-1, 20, 20), perc=99.5), num_rows=5))
#	imsave('results/c_hmoisa/data_rnd_phase.png',
#		stitch(imformat(data_rnd_phase.T.reshape(-1, 20, 20), perc=99.5), num_rows=5))
#	imsave('results/c_hmoisa/data_rnd_norm.png',
#		stitch(imformat(data_rnd_norm.T.reshape(-1, 20, 20), perc=99.5), num_rows=5))
#	imsave('results/c_hmoisa/data_rnd_sub.png',
#		stitch(imformat(data_rnd_sub.T.reshape(-1, 20, 20), perc=99.5), num_rows=5))
	imsave('results/c_hmoisa/data_knn_phase.png',
		stitch(imformat(data_knn_phase.T.reshape(-1, 20, 20), perc=99.5), num_rows=5))
#	imsave('results/c_hmoisa/data_knn_norm.png',
#		stitch(imformat(data_knn_norm.T.reshape(-1, 20, 20), perc=99.5), num_rows=5))
#	imsave('results/c_hmoisa/samples.png',
#		stitch(imformat(samples.T.reshape(-1, 20, 20), perc=99.5), num_rows=5))

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
