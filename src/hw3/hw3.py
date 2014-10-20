#
# hw3.py
#
# Running the experiments for hw3
#

from __future__ import division
import numpy as np
import ica_tools as ica
import scipy.io as io
#import scikits.audiolab as audio

data_dir = "dat"

def get_matlab_array(fname):
    return io.loadmat(fname)[fname[4:-4]]

def do_source_separation(mixes):

    # play start files
    # for mix in mixes:
    #     audio.play(mix, fs=44100)

    data = np.matrix(np.vstack(mixes))
    rotation5 = np.array([[ 0.39773973, -0.24205259, -0.15788047, -0.87079119 , 0.00317907],
                          [-0.44861537, -0.54780395 , 0.67873794 ,-0.1753857,  0.08491022],
                          [0.40412443,  0.57751707 , 0.69952414 ,-0.10256377, 0.0574816 ],
                          [0.11984375, -0.07139445, -0.07753464 , 0.09223065 , 0.98286428],
                          [0.68034424, -0.55018018 , 0.13799612 , 0.43810585, -0.1531464 ]])
    data = rotation5.dot(data)

    rotations = ica.transform(data)
    print rotations
    final_rotation_matrix = reduce(np.dot, reversed(rotations))
    print "final rotation matrix:"
    print final_rotation_matrix

    # play separated files
    # for i in range(len(mixes)):
    #     audio.play(data[i,:], 44100)

def do_source_sep3():
    print "--- 3 source separation ---"
    fnames = ["%s/mix%d.mat" % (data_dir,i) for i in range(1,4)]
    mixes = [io.loadmat(fname)[fname[4:-4]] for fname in fnames]
    do_source_separation(mixes)

def do_source_sep5():
    print "--- 5 source separation ---"
    fname = "%s/mixFive.mat" % (data_dir)
    mixes = [io.loadmat(fname)["mixFive%d" % (i)] for i in range(1,6)]
    do_source_separation(mixes)

# do source separation on 3 sources
#do_source_sep3()

# do source separation on 5 sources
do_source_sep5()