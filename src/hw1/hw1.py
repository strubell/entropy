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
import sys
import time

# a reasonable range for values of sigma: num_sigmas evenly spaced
# from min diff between samples to max diff between samples
def get_S(xs_sorted, n):
    num_sigmas = 10
    max_diff = xs_sorted[-1] - xs_sorted[0]
    min_diff = np.min([xs_sorted[i+1]-xs_sorted[i] for i in range(n-1)])
    return np.linspace(min_diff, max_diff, num_sigmas)

def estimate(rfunc):
    trials = 10
    ns = [10, 100, 1000]
    for n in ns:
        # generate some points
        xs = [rfunc(size=n) for i in range(trials)]
        for i in range(trials):
            xs[i].sort()

        # get range of sigmas to try, find best
        sigmas = [de.findBestSigma(xs[i], get_S(xs[i], n)) for i in range(trials)]

        # do Monte-Carlo estimate
        start = time.clock()
        N = 10000
        mc_estimates = [de.samplingEntropyEst(xs[i], N, sigmas[i]) for i in range(trials)]
        elapsed = time.clock()-start
        print "MC estimate %d samples (%d iters): mean %g, std %g (%fms)" % \
              (n, N, np.mean(mc_estimates), np.std(mc_estimates), elapsed/trials)

        # do m-spacings estimate
        start = time.clock()
        ms_estimates = [de.mspacingsEntropyEst(xs[i]) for i in range(trials)]
        elapsed = time.clock()-start
        print "m-spacings estimate %d samples: mean %g, std %g (%fms)" % \
              (n, np.mean(ms_estimates), np.std(ms_estimates)**2, elapsed/trials)
        sys.stdout.flush()


# for each distribution,
# for each n in [10, 100, 1000, 10000]
# for 10 samples
# 1. Find best sigma using plug-in estimate
# 2. Do Monte-Carlo estimation
# 3. Do m-spacings estimation

print "uniform(0,1) (lg(1) = 0.0)"
estimate(partial(np.random.uniform, low=0.0, high=1.0))
print

print "uniform(0,8) (lg(8) = 3.0)"
estimate(partial(np.random.uniform, low=0.0, high=8.0))
print

print "uniform(0,0.5) (lg(0.5) = -1.0)"
estimate(partial(np.random.uniform, low=0.0, high=0.5))
print

print "gaussian(0,1) (0.5*lg(2*pi*e) = 2.05)"
estimate(partial(np.random.normal, loc=0.0, scale=1.0))
print

print "gaussian(0,100) (0.5*lg(2*pi*e*100^2) = 8.69)"
estimate(partial(np.random.normal, loc=0.0, scale=100.0))
print

print "exponential(0,1) (1-lg(1) = 1.0)"
estimate(partial(np.random.exponential, scale=1.0))
print

print "exponential(0,100) (1-lg(100) = -5.64)"
estimate(partial(np.random.exponential, scale=100.0))
print