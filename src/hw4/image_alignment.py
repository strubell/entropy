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

# RGB value [0,255] with which to pad shifted matrices
PAD_VALUE = 0

# max number of pixes by which to shift image
MAX_SHIFT = 2
single_dir_shifts = range(1, MAX_SHIFT+1)
all_shifts = list(itertools.product(single_dir_shifts, single_dir_shifts, single_dir_shifts, single_dir_shifts))

def get_shift(shift, matrix):
    return np.pad(matrix, ((shift[0],shift[1]),(shift[2],shift[3])), mode='constant', constant_values=PAD_VALUE)[shift[1]:-shift[0], shift[3]:-shift[2]]

# compute the entropy of the discrete probability distribution x
def entropy(x):
    x_flat = x.flatten()
    return -np.sum((np.log2(x_i) if x_i > 0.0 else 0.0)*x_i for x_i in x_flat)

# compute the mutual information of the given joint distribution
# over two variables x and y:
# I(x,y) = sum_{x,y}p(x,y)log(p(x,y)/p(x)p(y))
# this is just the weighted average of the pointwise mutual infos
def mutInfo(joint):
    marginals_x = np.sum(joint, axis=0)
    marginals_y = np.sum(joint, axis=1)
    return entropy(marginals_x) + entropy(marginals_y) - entropy(joint)

# return proportion of values <= bin i for given number of bins
def distributionFromImage(layer, bins):
    bins, _ = np.histogram(layer, bins=bins, normed=True)
    return bins/np.sum(bins)

# return joint dist over bins (see distributionFromImage) for two images
def jointDistFromImages(layer1, layer2, bins):
    histo = np.histogram2d(layer2.flatten(), layer1.flatten(), bins=bins, normed=True)[0]
    return histo/np.sum(histo)

def getMutualInfoForJointDists(shift, layer1, layer2, bins):
    return mutInfo(jointDistFromImages(layer1, get_shift(shift, layer2), bins)), shift

def getBestShift(base, layer, bins):
    # all_shifted = [np.pad(layer, ((shift[0],shift[1]),(shift[2],shift[3])), mode='constant', constant_values=PAD_VALUE)[shift[1]:-shift[0], shift[3]:-shift[2]] for shift in all_shifts]
    print "computing mutual information"
    mutual_infos_shifts = zip(*pool.map(partial(getMutualInfoForJointDists, layer1=base, layer2=layer, bins=bins), all_shifts))
    print mutual_infos_shifts[0]
    best_shift = mutual_infos_shifts[1][np.argmax(mutual_infos_shifts[0])]
    return get_shift(best_shift, layer)

# only consider shifts in a single direction
# def getBestShiftSimple(base, layer, bins):
#     top_shifted = [np.pad(layer, ((i,0),(0,0)), mode='constant', constant_values=PAD_VALUE)[:-i, :] for i in range(1,MAX_SHIFT+1)]
#     bottom_shifted = [np.pad(layer, ((0,i),(0,0)), mode='constant', constant_values=PAD_VALUE)[i:, :] for i in range(1,MAX_SHIFT+1)]
#     left_shifted = [np.pad(layer, ((0,0),(i,0)), mode='constant', constant_values=PAD_VALUE)[:, :-i] for i in range(1,MAX_SHIFT+1)]
#     right_shifted = [np.pad(layer, ((0,0),(0,i)), mode='constant', constant_values=PAD_VALUE)[:, i:] for i in range(1,MAX_SHIFT+1)]
#     all_shifted = top_shifted + bottom_shifted + left_shifted + right_shifted
#     mutual_infos = pool.map(partial(getMutualInfoForJointDists, layer2=base, bins=bins), all_shifted)


def align(img, bins):
    layers = np.split(img[1:],3,axis=0)
    base = layers[0]
    bestShift1 = getBestShift(base, layers[1], bins)
    bestShift2 = getBestShift(base, layers[2], bins)
    # return np.transpose(np.array([base, bestShift1, bestShift2]), (1, 2, 0))
    return np.transpose(np.array([bestShift2, bestShift1, base]), (1, 2, 0))

pool = multiprocessing.Pool()
