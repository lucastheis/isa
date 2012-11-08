from numpy import max, min, hstack, zeros, eye, dot, sign, sqrt
from sqrtm import sqrtmi
from patches import imsave, imformat, stitch

def isavis(A, dims, num_cols=None):
	if num_cols is None:
		num_cols = max(dims) / 2

	m = min(A)
	k = 0
	subsp = []

	for dim in dims:
		while dim > num_cols:
			subsp.append(A[:, k:k + num_cols])
			dim -= num_cols
			k += num_cols
		subsp.append(hstack([A[:, k:k + dim], zeros([A.shape[0], num_cols - dim]) + m]))
		k += dim

	return hstack(subsp), num_cols



def localize(B, num_steps=10000, step_width=1e-3, visualize=False):
	W = eye(B.shape[1])

	patch_size = int(sqrt(B.shape[0]) + 0.5)

	for i in range(num_steps):
		# gradient step
		dW = dot(B.T , sign(dot(B, W)))
		W = W - step_width * dW

		# orthogonalize
		W = dot(sqrtmi(dot(W, W.T)), W)

		if visualize and not (i % 100):
			imsave('results/localize/basis.{0}.png'.format(i),
				stitch(imformat(dot(B, W).T.reshape(-1, patch_size, patch_size))))

	return dot(B, W)
