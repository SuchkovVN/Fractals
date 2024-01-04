import utils
import sys
import multiprocessing
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import core
import math
from anim import julia_set_anim, julia_set_anim_gpu

mpl.rcParams['animation.ffmpeg_path'] = r'win\\ffmpeg.exe'


coef = (-0.3 -0.2j)
maxIter = 500
radius = 2.5
def iterate(zs):
    res = np.zeros(zs.shape[0])
    
    i = 0
    for zp in zs:
        z = zp
        k = 0
        while abs(z) <= radius and k <= maxIter:
            z = z**2 + coef
            k += 1
        res[i] = k / maxIter
        i += 1
    
    return res

def iterate_z(cs):
    res = np.zeros(cs.shape[0])
    
    i = 0
    for c in cs:
        z = c
        k = 0
        while abs(z) <= 2 and k <= maxIter:
            z = z**2 + c
            k += 1
        res[i] = k / maxIter
        i += 1
    
    return res

def burning_ship(cs):
    res = np.zeros(cs.shape[0])
    
    i = 0
    for c in cs:
        z = c
        k = 0
        while abs(z) <= 2 and k <= maxIter:
            z = (complex(abs(z.real), abs(z.imag)))**2 + c
            k += 1
        res[i] = k / maxIter
        i += 1
    
    return res

def main(fname, set, procs):
    cfg = []
    with open(fname) as f:
        for line in f:
            cfg.append(float(line))
            
    print(cfg)
    xmin = cfg[0]
    xmax = cfg[1]
    ymin = cfg[2]
    ymax = cfg[3]
    xwidth = xmax - xmin
    yheight = ymax - ymin
    maxIter = int(cfg[4])
    im_width = int(cfg[5])
    im_height = int(cfg[6])
    net = (xmin, ymin, xwidth / im_width, yheight / im_height, im_width, im_height)
        
    start = time.monotonic()
    if set == 'julia':
        mapp = utils.julia_cmap_parallel(iterate, net, procs)
    elif set == 'mbrot':
        mapp = utils.mbrot_cmap_parallel(iterate_z, net, procs)
        mapp = np.transpose(mapp)
    elif set == 'mbrot_numba':
        mapp = core.mbrot_set_f((xmin, xmax), (ymin, ymax), (im_width, im_height), maxIter)
    elif set == 'julia_numba':
        mapp = core.julia_set_f((xmin, xmax), (ymin, ymax), (im_width, im_height), maxIter, 0.1, 0.65)
    elif set == 'mbrot_numba_gpu':
        mapp = np.zeros((im_width, im_height))
        threadsperblock = (32, 32)
        blockspergrid_x = math.ceil(mapp.shape[0] / threadsperblock[0])
        blockspergrid_y = math.ceil(mapp.shape[1] / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        core.mbrot_set_f_gpu[blockspergrid, threadsperblock](mapp, (xmin, xmax, ymin, ymax, im_width, im_height, maxIter))
    elif set == 'julia_anim':
        anim_fps = int(cfg[7])
        duration = int(cfg[8])
        anglmin = float(cfg[9])
        anglmax = float(cfg[10])
        start_r = float(cfg[11])
        N = int(cfg[12])
        spiral_speed = float(cfg[13])
        coefs = utils.generate_spiral_seq(start_r, (anglmin, anglmax), N, spiral_speed)

        julia_set_anim((xmin, xmax), (ymin, ymax), (im_width, im_height), maxIter, coefs, N, anim_fps, duration)
    elif set == 'julia_anim_gpu':
        anim_fps = int(cfg[7])
        duration = int(cfg[8])
        anglmin = float(cfg[9])
        anglmax = float(cfg[10])
        start_r = float(cfg[11])
        N = int(cfg[12])
        spiral_speed = float(cfg[13])
        coefs = utils.generate_spiral_seq(start_r, (anglmin, anglmax), N, spiral_speed)

        julia_set_anim_gpu(xmin, xmax, ymin, ymax, im_width, im_height, maxIter, coefs, N, anim_fps, duration)
    else:
        print(f"Error: unsupported set {set}")
        return
    stop = time.monotonic() - start
    print(f"Elapsed time: {stop}s")
        
    if set != 'julia_anim' and set != 'julia_anim_gpu':
        fig, ax = plt.subplots()
        plt.imsave(fname=f"images/{set}_{im_width}x{im_height}s.png", arr=mapp, cmap="flag_r", vmin=0, vmax=2.25)
        plt.imshow(mapp, cmap="flag_r", vmin=0, vmax=2.25)
        
        xtick_labels = np.linspace(xmin, xmax, int(xwidth * 2))
        ax.set_xticks([(x-xmin) / xwidth * im_width for x in xtick_labels])
        ax.set_xticklabels(['{:.1f}'.format(xtick) for xtick in xtick_labels])
        ytick_labels = np.linspace(ymin, ymax, int(yheight * 2))
        ax.set_yticks([(y-ymin) / yheight * im_height for y in ytick_labels])
        ax.set_yticklabels(['{:.1f}'.format(ytick) for ytick in ytick_labels])
        
        plt.show()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))