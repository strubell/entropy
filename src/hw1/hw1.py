#
# hw1.py
#
# Answering the questions for hw1
#
#
from __future__ import division
import differential_entropy as de
from functools import partial
import numpy as np
import time

# a reasonable range for values of sigma: num_sigmas evenly spaced
# from min diff between samples to max diff between samples
def get_S(xs_sorted, n):
    num_sigmas = 10
    max_diff = xs_sorted[-1] - xs_sorted[0]
    min_diff = np.min([xs_sorted[i+1]-xs_sorted[i] for i in range(n-1)])
    return np.linspace(min_diff, max_diff, num_sigmas)

def estimate(rfunc):
    trials = 1
    ns = [10, 100, 1000]
    for n in ns:
        # generate some points
        xs = rfunc(size=n)
        xs_sorted = sorted(xs)

        # get range of sigmas to try, find best
        S = get_S(xs_sorted, n)
        sigma = de.findBestSigma(xs, S)
        print "Using sigma = %g" % (sigma)

        # # do Monte-Carlo estimate
        # start = time.clock()
        # N = 10000
        # mc_estimates = [de.samplingEntropyEst(xs, N, sigma) for i in range(trials)]
        # elapsed = time.clock()-start
        # print "MC estimate %d samples (%d iters): mean %g, std %g (%dms)" % \
        #       (n, N, np.mean(mc_estimates), np.std(mc_estimates), elapsed)

        # do m-spacings estimate
        start = time.clock()
        ms_estimates = [de.mspacingsEntropyEst(xs_sorted) for i in range(trials)]
        elapsed = time.clock()-start
        print "m-spacings estimate %d samples: mean %g, std %g (%dms)" % \
              (n, np.mean(ms_estimates), np.std(ms_estimates)**2, elapsed)


# for each distribution,
# for each n in [10, 100, 1000, 10000]
# for 10 samples
# 1. Find best sigma using plug-in estimate
# 2. Do Monte-Carlo estimation
# 3. Do m-spacings estimation

# print "uniform(0,1)"
# uniform_01 = partial(np.random.uniform, low=0.0, high=1.0)
# estimate(uniform_01)

print "uniform(0,8)"
uniform_08 = partial(np.random.uniform, low=0.0, high=8.0)
estimate(uniform_08)