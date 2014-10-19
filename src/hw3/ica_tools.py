#
# ica_tools.py
#
# Tools for Independent Components Analysis (ICA)
#
from __future__ import division
import numpy as np
import itertools
from functools import partial
import multiprocessing

'''
m-spacings estimation:
discretize range into bins with width = round(sqrt(n))
'''
def mspacingsEntropyEst(xs):
    xs_sorted = np.sort(xs[0])
    n = xs_sorted.shape[1]
    m = int(round(np.sqrt(n)))
    things = [((n+1)/m)*(xs_sorted[0,i+m]-xs_sorted[0,i]) for i in range(n-m)]
    return np.mean(np.log2(things))

def get_rotation_mat(theta, dim, p, q):
    ret = np.eye(dim)
    ret[p,p] = np.cos(theta)
    ret[q,q] = np.cos(theta)
    ret[p,q] = -np.sin(theta)
    ret[q,p] = np.sin(theta)
    return ret

def entropy_est(rot):
    return np.sum([mspacingsEntropyEst(rot[i]) for i in range(len(rot))])

def find_best_rotation(data, rotations):
    rotated_data = [rot.dot(data) for rot in rotations]
    return rotations[np.argmin(pool.map(entropy_est, rotated_data))]

def transform(data):
    dim = data.shape[0]
    thetas = np.linspace(0, np.pi, 180)
    directions = list(itertools.combinations(range(dim), 2))
    num_directions = len(directions)
    rotation_matrices = [map(partial(get_rotation_mat, dim=dim, p=direction[0], q=direction[1]), thetas) for direction in directions]
    best_rotations = np.empty((2*num_directions, dim, dim))
    for i in range(2*num_directions):
        print "getting best rotation in direction %s" % str(directions[i % num_directions])
        best_rotations[i] = find_best_rotation(data, rotation_matrices[i % num_directions])
        data = best_rotations[i].dot(data)
    return best_rotations

pool = multiprocessing.Pool()
