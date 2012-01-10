__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@tuebingen.mpg.de>'
__docformat__ = 'epytext'

from numpy import log, array, asarray, mean, std
from numpy.random import randn, permutation
from numpy.linalg import eig

def preprocess(data, shuffle=True, noise_level=None):
	"""
	Log-transforms and centers the data. Optionally, adds some noise.
	The standard deviation of the added Gaussian noise is 1 / C{noise_level}.

	@type  data: array_like
	@param data: data points stored in columns

	@type  shuffle: boolean
	@param shuffle: whether or not to randomize the order of the data

	@type  noise_level: integer
	@param noise_level: add a little bit of noise after log-transform

	@rtype: ndarray
	@return: preprocessed data
	"""

	data = array(data, dtype='float64')

	# log-transform
	data[data < 1.] = 1.
	data = log(data)

	# randomize order
	if shuffle:
		data = data[:, permutation(data.shape[1])]

	# center
	data = data - mean(data, 1).reshape(-1, 1)

	if noise_level is not None:
		# add Gaussian white noise
		data += randn(*data.shape) * (std(data) / float(noise_level))

	return asarray(data, order='F')
