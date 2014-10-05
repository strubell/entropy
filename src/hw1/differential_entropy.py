#
# differential_entropy.py
#
# Some functions for estimating the differential entropy given a 
# bunch of observed data points.
#
from __future__ import division
import numpy as np
import scipy.stats as stats

'''
Leave-one-out estimation
'''
# Takes the original sample points xs organized as a column vector, the point x at which
# the density is to be evaluated, and the standard deviation sigma of the kernel, and
# returns p, the density of the kernel density estimate at the point x.
def densEst(xs, x, sigma):
    # print "mu: %g, sigma: %g" % (x, sigma)
    # return np.mean([normal_pdf(xs[i], x, sigma) for i in range(len(xs))])
    return np.mean([stats.norm.pdf(xs[i], x, sigma) for i in range(len(xs))])

# Generates a leave-one-out point estimate for data point j using densEst
def looPoint(xs, j, sigma):
    xs_less_j = [x for idx, x in enumerate(xs) if not idx == j]
    return densEst(xs_less_j, xs[j], sigma)

def looAvgLogLike(xs, sigma):
    looPoints = [looPoint(xs, j, sigma) for j in range(xs.size)]
    return np.mean(np.log([x for x in looPoints if x != 0])), looPoints

def findBestSigma(xs, S):
    looAvgLogLikes, points = zip(*[looAvgLogLike(xs, sigma) for sigma in S])
    return np.mean(points[np.argmax(looAvgLogLikes)])

'''
Monte-Carlo estimation
'''
# Choose a random point from xs, add random Gaussian offset
def sampleNPDE(xs, sigma):
    point = xs[np.random.randint(0, xs.size)]
    offset = np.random.randn()*sigma
    return point + offset

def samplingEntropyEst(xs, N, sigma):
    return np.mean(np.log([densEst(xs, sampleNPDE(xs, sigma), sigma) for i in range(N)]))

'''
m-spacings estimation
'''
# discretize range into bins with width = round(sqrt(n))
def mspacingsEntropyEst(xs):
    return xs