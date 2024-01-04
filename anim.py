import utils
import sys
import multiprocessing
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import core
import math
from numba import njit, prange
from matplotlib import animation, rc

def julia_set_anim(xlims, ylims, im_dims, maxIter, coefs, N, fps, nSeconds):
    mapp = core.julia_set_f(xlims, ylims, im_dims, maxIter, coefs[0, 0], coefs[0, 1])

    dpi = 100

    fig = plt.figure(frameon=False)
    fig.set_size_inches(im_dims[0] / dpi, im_dims[1] / dpi)
    plt.axis('off')
    image = plt.imshow(mapp, cmap="flag_r", vmin=0, vmax=2.25, aspect='auto')

    # initialization function: plot the background of each frame
    def init():
        return [image]

    # animation function.  This is called sequentially
    def animate(i):
        mapp = core.julia_set_f(xlims, ylims, im_dims, maxIter, coefs[i, 0], coefs[i, 1])
        image.set_array(mapp)
        return [image]

    anim = animation.FuncAnimation(
                               fig, 
                               animate, 
                               frames = nSeconds * fps,
                               interval = 1000 / fps, # in ms
                               )

    FFwriter = animation.FFMpegWriter(fps=fps)
    anim.save(f"images/julia_{im_dims[0]}s_set.mp4", writer=FFwriter)

def julia_set_anim_gpu(xmin, xmax, ymin, ymax, im_w, im_h, maxIter, coefs, N, fps, nSeconds):
    mapp = np.zeros((im_w, im_h)) 

    dpi = 100

    xlims = (xmin, xmax)
    ylims = (ymin, ymax)
    im_dims = (im_w, im_h)

    threadsperblock = (32, 32)
    blockspergrid_x = math.ceil(mapp.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(mapp.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    core.julia_set_f_gpu[blockspergrid, threadsperblock](mapp, xmin, xmax, ymin, ymax, im_w, im_h, maxIter, coefs[0, 0], coefs[0, 1])

    fig = plt.figure(frameon=False)
    fig.set_size_inches(im_dims[0] / dpi, im_dims[1] / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    image = plt.imshow(mapp, cmap="flag_r", vmin=0, vmax=2.25, aspect='auto')

    # initialization function: plot the background of each frame
    def init():
        return [image]

    # animation function.  This is called sequentially
    def animate(i):
        core.julia_set_f_gpu[blockspergrid, threadsperblock](mapp, xmin, xmax, ymin, ymax, im_w, im_h, maxIter, coefs[i, 0], coefs[i, 1])
        image.set_array(mapp)
        return [image]

    anim = animation.FuncAnimation(
                               fig, 
                               animate, 
                               frames = nSeconds * fps,
                               interval = 1000 / fps, # in ms
                               )


    FFwriter = animation.FFMpegWriter(fps=fps)
    anim.save(f"images/julia_{im_dims[0]}s_gpu_set.mp4", writer=FFwriter)