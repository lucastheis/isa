import sys

sys.path.append('./code')

from tools import Experiment, logmeanexp
from numpy import log, mean, std, sqrt

def main(argv):
	if len(argv) < 3:
		print 'Usage:', argv[0], '<dim> <experiment>'

	dim = int(argv[1])

	for i in range(2, len(argv)):
		results = Experiment(argv[i])

		if 'fixed' in results.results:
			print 'ALREADY FIXED'
			print
			continue

		print 'BEFORE:'
		print results['loglik'], ' +- ', results['sem']
		print


		results['ais_weights'] = results['ais_weights'] * log(2.) * dim
		results['loglik'] = mean(logmeanexp(results['ais_weights'], 0))
		results['loglik'] = results['loglik'] / log(2.) / dim
		results['sem'] = std(logmeanexp(results['ais_weights'], 0), ddof=1)
		results['sem'] = results['sem'] / sqrt(results['ais_weights'].shape[1]) / log(2.) / dim 

		print 'AFTER:'
		print results['loglik'], ' +- ', results['sem']
		print

		results['fixed'] = True
		results.save(overwrite=True)

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
