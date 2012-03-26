import sys

sys.path.append('./code')

from tools import Experiment
from time import sleep

def main(argv):
	experiment = Experiment(server='newton')

	for i in range(10):
		experiment.progress(i * 10)
		experiment['number'] = i
		experiment.save('/kyb/agmb/lucas/.Trash/test.xpck')
		sleep(1)

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
