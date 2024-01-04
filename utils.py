import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import time
import math
from numba import njit, prange

# net = (x_start, y_start, x_step, y_step, x_dim, y_dim)

def make_iter(c, maxIter, radius):
    def iterate(z):
        k = 0
        while abs(z) <= radius and k <= maxIter:
            z = z**2 + c
            k += 1
        return k / maxIter
    return iterate
    
def julia_cmap_parallel(rhs, net, procs):
    map = np.zeros((net[4], net[5]))
    
    x = net[0]
    y = net[1]
    i = 0
    while i < map.shape[0] - procs:
        zs = [None] * procs
        
        res = []
        for d in range(0, procs):
            zs[d] = np.zeros(map.shape[1], dtype=complex)
            for j in range(0, zs[0].shape[0]):
                zs[d][j] = complex(x + (i + d) * net[2], y + j * net[3])
        
        with multiprocessing.Pool(procs) as pool:
            res = pool.map(rhs, zs)
        
        for r in res:
            for j in range(0, map.shape[1]):
                map[i, j] = r[j]
            i += 1
        
    if map.shape[0] - i > 0:
        dr = map.shape[0] - i
        zs = [None] * dr
        
        res = []
        for d in range(0, dr):
            zs[d] = np.zeros(map.shape[1], dtype=complex)
            for k in range(0, map.shape[1]):
                zs[d][k] = complex(x + (i + d) * net[2], y + k * net[3])
        
        with multiprocessing.Pool(dr) as pool:
            res = pool.map(rhs, zs)
        
        for r in res:
            for j in range(0, map.shape[1]):
                map[i, j] = r[j]
            i += 1

    return map


def mbrot_cmap_parallel(rhs, net, procs):
    map = np.zeros((net[4], net[5]))
    
    x = net[0]
    y = net[1]
    j = 0
    while j < map.shape[1] - procs:
        cs = [None] * procs
        
        res = []
        for d in range(0, procs):
            cs[d] = np.zeros(map.shape[0], dtype=complex)
            for i in range(0, cs[0].shape[0]):
                cs[d][i] = complex(x + i * net[2], y + (j + d) * net[3])
        
        with multiprocessing.Pool(procs) as pool:
            res = pool.map(rhs, cs)
        
        for r in res:
            for i in range(0, map.shape[0]):
                map[i, j] = r[i]
            j += 1
        
    if map.shape[0] - j > 0:
        dr = map.shape[0] - j
        cs = [None] * dr
        
        res = []
        for d in range(0, dr):
            cs[d] = np.zeros(map.shape[0], dtype=complex)
            for k in range(0, map.shape[0]):
                cs[d][k] = complex(x + k * net[2], y + (j + d) * net[3])
        
        with multiprocessing.Pool(dr) as pool:
            res = pool.map(rhs, cs)
        
        for r in res:
            for i in range(0, map.shape[0]):
                map[i, j] = r[i]
            j += 1

    return map

@njit(parallel=True)
def generate_rot_seq(r, tlims, n):
    res = np.zeros((n, 2))
    step = (tlims[1] - tlims[0]) / n

    for i in prange(n):
        t = tlims[0] + i * step
        res[i, 0] = r * math.cos(t)
        res[i, 1] = r * math.sin(t)

    return res

@njit(parallel=True)
def generate_spiral_seq(r_start, tlims, n, sp):
    res = np.zeros((n, 2))
    step = (tlims[1] - tlims[0]) / n

    for i in prange(n):
        t = tlims[0] + i * step
        r = r_start * math.exp(-sp * t)
        res[i, 0] = r * math.cos(t)
        res[i, 1] = r * math.sin(t)

    return res


