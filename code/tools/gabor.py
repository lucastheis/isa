from numpy import meshgrid, linspace, pi, exp, square, sin, cos, abs, sqrt
from numpy import min, max, arctan2, argmax
from numpy.random import rand
from numpy.fft import fft2, fftshift

def gaborf(size, complex=True, f=None, a=None, s=None, t=None, x=None, y=None, p=None):
	"""
	Creates a 2D Gabor filter. If size is a square integer, a one-dimensional
	array containing the flattened Gabor filter is returned.

	@type  size: int/tuple
	@param size: square integer or tuple determining image size

	@type  complex: boolean
	@param complex: whether to return both real and imaginary parts

	@type  f: float
	@param f: frequency

	@type  a: float
	@param a: aspect ratio

	@type  s: float
	@param s: scale

	@type  t: float
	@param t: angle

	@type  x: float
	@param x: position in horizontal direction

	@type  y: float
	@param y: position in vertical direction

	@type  p: float
	@param p: phase

	@rtype: array
	@return: Gabor filter
	"""

	if isinstance(size, int):
		if abs(sqrt(size) - int(sqrt(size))) > 1E-10:
			raise RuntimeError('Filter size must be square.')

		scale = sqrt(size) / 16.
		xx, yy = meshgrid(
			linspace(-5 * scale, 5 * scale, sqrt(size)),
			linspace(-5 * scale, 5 * scale, sqrt(size)))
	else:
		xx, yy = meshgrid(
			linspace(-5 * size[0] / 16., 5 * size[0] / 16., size[0]),
			linspace(-5 * size[1] / 16., 5 * size[1] / 16., size[1]))

	# angle
	if t is None:
		t = rand() * pi * 2.

	# aspect ratio
	if a is None:
		a = rand() / 2. + 0.75

	# scale
	if s is None:
		s = rand() / 1. + 2.

	# frequency
	if f is None:
		f = rand() * 2. + 2.

	# center
	if x is None:
		x = rand() * (max(xx) - min(xx)) + min(xx)

	if y is None:
		y = rand() * (max(yy) - min(yy)) + min(yy)

	# phase
	if p is None:
		p = rand() * pi * 2.

	xx, yy = xx - x, yy - y
	xx, yy = xx * cos(t) + yy * sin(t), -xx * sin(t) + yy * cos(t)

	G = exp(-square(xx) / s - square(a * yy) / s) * cos(f * xx)

	if complex:
		G = G + 1j * exp(-square(xx) / s - square(a * yy) / s) * sin(f * xx + p)

	if isinstance(size, int):
		return G.flatten()
	return G



def gaborfit(gabor):
	"""
	Find position and orientation of Gabor filters.
	"""

	if gabor.ndim < 2:
		gabor = gabor.reshape(int(sqrt(gabors.shape[0]) + 0.5), -1)

	y, Y = argmax(abs(gabor), 0), max(abs(gabor), 0)
	x = argmax(Y)
	y = y[x]

	x = x - gabor.shape[0] / 2. + 1.
	y = y - gabor.shape[1] / 2. + 1.
	x = x * 10. / 16.
	y = y * 10. / 16.

	f = fftshift(fft2(gabor))

	m, M = argmax(abs(f), 0), max(abs(f), 0)
	n = argmax(M)
	m = m[n]

	m = m - gabor.shape[0] / 2
	n = n - gabor.shape[1] / 2

	t = arctan2(m, n)

	return x, y, t
