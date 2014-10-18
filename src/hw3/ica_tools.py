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
    things = [((n+1)/m)*(xs_sorted[0,i+m]-xs_sorted[0,i]) for i in range(0, n-m)]
    return np.mean(np.log2(things))

def get_rotation_mat(theta, dim, p, q):
    ret = np.eye(dim)
    ret[p,p] = np.cos(theta)
    ret[q,q] = np.cos(theta)
    ret[p,q] = -np.sin(theta)
    ret[q,p] = np.sin(theta)
    return np.matrix(ret)

def entropy_est(rot):
    return np.sum([mspacingsEntropyEst(rot[i,:]) for i in range(len(rot))])

def find_best_rotation(data, rotations):
    # return rotations[np.argmin([np.sum(np.apply_along_axis(mspacingsEntropyEst, 1, rot*data)) for rot in rotations])]
    print "\tcomputing rotation matrices"
    rotated_data = [rot*data for rot in rotations]
    print "\tfinding min entropy rotation"
    # sums = [np.sum([mspacingsEntropyEst(rot[i,:]) for i in range(len(rot))]) for rot in rotated_data]
    sums = pool.map(entropy_est, rotated_data)
    print sums
    return rotations[np.argmin(sums)]

# a single iteration of finding best rotation in each direction
# and applying it to the data, returns the rotation matrices used
def transform(data):
    dim = data.shape[0]
    thetas = np.linspace(0, 2*np.pi, 10)
    directions = itertools.combinations(range(dim), 2)
    print "getting rotation matrices"
    rotation_matrices = [map(partial(get_rotation_mat, dim=dim, p=direction[0], q=direction[1]), thetas) for direction in directions]
    best_rotations = np.empty((dim*(dim-1)/2, dim, dim))
    for i in range(dim):
        print "getting best rotation in direction %d" % (i)
        best_rotations[i] = find_best_rotation(data,rotation_matrices[i])
        print "best rotation:"
        print best_rotations[i]
        data = best_rotations[i] * data
    return best_rotations

pool = multiprocessing.Pool()
