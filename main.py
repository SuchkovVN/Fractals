import utils
import sys
import multiprocessing
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

coef = (-0.3 -0.2j)
maxIter = 900
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

def main(procs):
    xmin = -1.5
    xmax = 0.5
    ymin = -1.1
    ymax = 1.1
    xwidth = xmax - xmin
    yheight = ymax - ymin
    im_width = 512
    im_height = 512
    net = (xmin, ymin, xwidth / im_width, yheight / im_height, im_width, im_height)
        
    start = time.clock_gettime(0)
    # jset = utils.cmap_parallel(iterate, net, maxIter, coef, 2., 8)
    mbrot = utils.mbrot_cmap_parallel(iterate_z, net, procs)
    stop = time.clock_gettime(0) - start
    print(f"Elapsed time: {stop}s")
        
    fig, ax = plt.subplots()
    mbrot = np.transpose(mbrot)
    # twilight_shifted for julia's set
    plt.imsave(fname="mbrot2.png", arr=mbrot, cmap="flag_r", vmin=0, vmax=2.25)
    plt.imshow(mbrot, cmap="flag_r", vmin=0, vmax=2.25)
    
    xtick_labels = np.linspace(xmin, xmax, int(xwidth * 2))
    ax.set_xticks([(x-xmin) / xwidth * im_width for x in xtick_labels])
    ax.set_xticklabels(['{:.1f}'.format(xtick) for xtick in xtick_labels])
    ytick_labels = np.linspace(ymin, ymax, int(yheight * 2))
    ax.set_yticks([(y-ymin) / yheight * im_height for y in ytick_labels])
    ax.set_yticklabels(['{:.1f}'.format(ytick) for ytick in ytick_labels])
    
    plt.show()


if __name__ == "__main__":
    main(int(sys.argv[1]))