#
# differential_entropy.py
#
# Some functions for estimating the differential entropy given a 
# bunch of observed data points.
#
from __future__ import division
import multiprocessing
import numpy as np
from functools import partial

'''
Leave-one-out estimation
'''
# Takes the original sample points xs organized as a column vector, the point x at which
# the density is to be evaluated, and the standard deviation sigma of the kernel, and
# returns p, the density of the kernel density estimate at the point x.
def densEst(xs, x, sigma):
    s = 1/(np.sqrt(2*np.pi)*sigma)
    return np.mean([s * np.exp(-((x-x_i)**2 / (2*sigma**2))) for x_i in xs])

# Generates a leave-one-out point estimate for data point j using densEst
def looPoint(j, xs, sigma):
    xs_less_j = np.append(xs[:j], xs[j+1:])
    return densEst(xs_less_j, xs[j], sigma)

def looAvgLogLike(xs, sigma):
    chunks = len(xs)//pool._processes
    looPoints = pool.map(partial(looPoint, xs=xs, sigma=sigma), range(xs.size), chunks)
    return np.mean(np.log2([x for x in looPoints if x != 0.0]))

# Returns the best sigma in S (the one that maximizes the log likelihood)
def findBestSigma(xs, S):
    return S[np.argmax([looAvgLogLike(xs, sigma) for sigma in S])]

'''
Monte-Carlo estimation
'''
# Choose a random point from xs, add random Gaussian offset
def sampleNPDE(xs, sigma):
    point = xs[np.random.randint(0, xs.size)]
    return np.random.normal(point, sigma)

def samplingEntropyEst(xs, N, sigma):
    samples = [densEst(xs, sampleNPDE(xs, sigma), sigma=sigma) for i in range(N)]
    return np.mean(-np.log2(samples))

'''
m-spacings estimation
'''
# discretize range into bins with width = round(sqrt(n))
def mspacingsEntropyEst(xs_sorted):
    n = len(xs_sorted)
    m = int(round(np.sqrt(n)))
    things = [((n+1)/m)*(xs_sorted[i+m]-xs_sorted[i]) for i in range(0, n-m)]
    return np.mean(np.log2(things))

pool = multiprocessing.Pool()