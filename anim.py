import utils
import sys
import multiprocessing
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from matplotlib import animation, rc

coef = (-0.3 -0.2j)
coefs = []
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

rc('animation', html='html5')
# отображать анимацию в виде html5 video

fig = plt.figure(figsize=(10, 10))
max_frames = 200

images = []
# кэш картинок

def init():
    return plt.gca()

net = ()
procs = 1

def animate_mbrot(i):
    if i > max_frames // 2:
        # фаза zoom out, можно достать картинку из кэша

        plt.imshow(images[max_frames//2-i], cmap="flag_r", vmin=0, vmax=2.25)
        return
    
    coef = coefs[i]
    image = utils.julia_cmap_parallel(iterate_z, net, procs)
    plt.imshow(image, cmap="flag_r", vmin=0, vmax=2.25)
    images.append(image)
    
    # добавить картинку в кэш
    return plt.gca()

def generate_coefs(start, stop, step):
    res = []
    
    


def main(cfgfname, fname, pr):
    cfg = []
    with open(cfgfname) as f:
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
    procs = pr
    
    params = []
    with open(fname) as f:
        for line in f:
            params.append(line)
            
        
    
    
    
    max_frames = len(coefs)
    animation.FuncAnimation(fig, animate_mbrot, init_func=init,
                                frames=max_frames, interval=50)

main(sys.argv[1], sys.argv[2], int(sys.argv[3]))