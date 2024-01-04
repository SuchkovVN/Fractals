import numpy as np 
from numba import njit, cuda, prange

@njit
def iterate_mbrot_f(ca, cb, maxIter):
	x = ca
	y = cb

	k = 0
	while (x**2 + y**2 < 4) and k <= maxIter:
		prev_x = x
		x = (x**2 - y**2 + ca)
		y = (2 * prev_x * y + cb)
		k += 1

	return k / maxIter

@njit
def iterate_z_f(zr, zi, ca, cb, maxIter):
	x = zr
	y = zi

	k = 0
	while (x**2 + y**2 < 4) and k <= maxIter:
		prev_x = x
		x = (x**2 - y**2 + ca)
		y = (2 * prev_x * y + cb)
		k += 1

	return k / maxIter


@njit(parallel=True)
def mbrot_set_f(xlims, ylims, im_dims, maxIter):
	x = xlims[0]
	y = ylims[0]

	xstep = (xlims[1] - xlims[0]) / im_dims[0] 
	ystep = (ylims[1] - ylims[0]) / im_dims[1]

	mapp = np.zeros((im_dims[1], im_dims[0]))
	for i in prange(mapp.shape[1]):
		for j in prange(mapp.shape[0]):
			mapp[i, j] = iterate_mbrot_f(x + j * xstep, y + i * ystep, maxIter)

	return mapp

@njit(parallel=True)
def julia_set_f(xlims, ylims, im_dims, maxIter, creal, cimag):
	x = xlims[0]
	y = ylims[0]

	xstep = (xlims[1] - xlims[0]) / im_dims[0] 
	ystep = (ylims[1] - ylims[0]) / im_dims[1]

	mapp = np.zeros((im_dims[1], im_dims[0]))
	for i in prange(mapp.shape[1]):
		for j in prange(mapp.shape[0]):
			mapp[i, j] = iterate_z_f(x + j * xstep, y + i * ystep, creal, cimag, maxIter)

	return mapp

# GPU

@cuda.jit
def mbrot_set_f_gpu(io_data, params):
	""" params = (xmin, xmax, ymin, ymax, im_w, im_h, maxIter), io_data = 2d array """

	x = params[0]
	y = params[2]

	i, j = cuda.grid(2)

	xstep = (params[1] - params[0]) / params[4] 
	ystep = (params[3] - params[2]) / params[5]

	io_data[i, j] = iterate_mbrot_f(x + j * xstep, y + i * ystep, params[6])

@cuda.jit
def julia_set_f_gpu(io_data, xmin, xmax, ymin, ymax, im_w, im_h, maxIter, creal, cimag):
	x = xmin
	y = ymin

	xstep = (xmax - xmin) / im_w 
	ystep = (ymax - ymin) / im_h

	i, j = cuda.grid(2)

	io_data[i, j] = iterate_z_f(x + j * xstep, y + i * ystep, creal, cimag, maxIter)



