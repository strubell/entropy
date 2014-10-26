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

# max number of pixes by which to shift image
MAX_SHIFT = 15

# RGB value [0,255] with which to pad shifted matrices
PAD_VALUE = 255

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
    return np.sum([joint[pair[0],pair[1]]*np.log2(joint[pair[0],pair[1]]/(marginals_x[pair[0]]*marginals_y[pair[1]])) for pair in pairs])

# return proportion of values <= bin i for given number of bins
def distributionFromImage(layer, bins):
    bins, _ = np.histogram(layer, bins=bins)
    return bins/np.sum(bins)

# return joint dist over bins (see distributionFromImage) for two images
def jointDistFromImages(layer1, layer2, bins):
    print np.sum(np.outer(distributionFromImage(layer1, bins), distributionFromImage(layer2, bins)))
    return np.outer(distributionFromImage(layer1, bins), distributionFromImage(layer2, bins))

def getMutualInfoForJointDists(layer1, layer2, bins):
    return mutInfo(jointDistFromImages(layer1, layer2, bins))

def getBestShift(base, layer, bins):
    top_shifted = [np.pad(layer, ((i,0),(0,0)), mode='constant', constant_values=PAD_VALUE)[:-i, :] for i in range(1,MAX_SHIFT+1)]
    bottom_shifted = [np.pad(layer, ((0,i),(0,0)), mode='constant', constant_values=PAD_VALUE)[i:, :] for i in range(1,MAX_SHIFT+1)]
    left_shifted = [np.pad(layer, ((0,0),(i,0)), mode='constant', constant_values=PAD_VALUE)[:, :-i] for i in range(1,MAX_SHIFT+1)]
    right_shifted = [np.pad(layer, ((0,0),(0,i)), mode='constant', constant_values=PAD_VALUE)[:, i:] for i in range(1,MAX_SHIFT+1)]
    all_shifted = top_shifted + bottom_shifted + left_shifted + right_shifted
    # mutual_infos = map(partial(getMutualInfoForJointDists, layer2=base, bins=bins), all_shifted)
    mutual_infos = pool.map(partial(getMutualInfoForJointDists, layer2=base, bins=bins), all_shifted)
    print mutual_infos
    return all_shifted[np.argmax(mutual_infos)]

def align(img, bins):
    layers = np.split(img[1:],3,axis=0)
    base = layers[0]
    bestShift1 = getBestShift(base, layers[1], bins)
    bestShift2 = getBestShift(base, layers[2], bins)
    return np.transpose(np.array([base, bestShift1, bestShift2]), (1, 2, 0))

pool = multiprocessing.Pool()
