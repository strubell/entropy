#
# image_alignment.py
#
# Tools for image alignment using mutual information
#
from __future__ import division
import numpy as np
import itertools
from functools import partial
import multiprocessing

# compute the entropy of the discrete probability distribution x
def entropy(x):
    zeroless = x[np.where(x != 0)]
    return -np.log2(zeroless).dot(zeroless)

# compute the mutual information of the given joint distribution
# over two variables x and y:
# I(x,y) = sum_{x,y}p(x,y)log(p(x,y)/p(x)p(y))
# this is just the weighted average of the pointwise mutual infos
def mutInfo(joint):
    marginals_x = np.sum(joint, axis=0)
    marginals_y = np.sum(joint, axis=1)
    num_x, num_y = np.shape(joint)
    pairs = itertools.product(range(num_x),range(num_y))
    return np.sum([joint[pair(0),pair(1)]*np.log2(joint[pair(0),pair(1)]/(marginals_x[pair(0)]*marginals_y[pair(1)])) for pair in pairs])

#def distributionFromImage(layer, bins):


pool = multiprocessing.Pool()
