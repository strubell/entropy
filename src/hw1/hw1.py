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
def get_S(xs, n):
    num_sigmas = 10
    diffs = [[np.abs(xs[i]-xs[j]) for i in range(n) if xs[i] != xs[j]] for j in range(n)]
    min_diff = np.min(diffs)
    max_diff = np.max(diffs)
    return np.linspace(min_diff, max_diff, num_sigmas)

def plugin_estimator(xs, S):
    return de.findBestSigma(xs, S)

def mc_estimator(xs, S):
    return de.findBestSigma(xs, S)

def estimate(rfunc, estimator):
    ns = [10, 100]
    for n in ns:
        xs = rfunc(size=n)
        S = get_S(xs, n)
        start = time.clock()
        estimates = [estimator(xs, S) for r in range(10)]
        elapsed = time.clock()-start
        print "%d samples: mean %g, std %g (%dms)" % (n, np.mean(estimates), np.std(estimates)**2, elapsed)

print "uniform(0,1)"
uniform_01 = partial(np.random.uniform, low=0.0, high=1.0)
print "plug-in estimate"
estimate(uniform_01, plugin_estimator)
print "monte carlo estimate"
estimate(uniform_01, plugin_estimator)