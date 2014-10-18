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
    rotations1 = ica.transform(data)
    rotations2 = ica.transform(data)
    rotations = np.concatenate((rotations1, rotations2))
    print rotations.shape
    final_rotation_matrix = reduce(np.dot, reversed(rotations))
    print "final rotation matrix:"
    print final_rotation_matrix

    # play separated files
    # for i in range(len(mixes)):
    #     audio.play(data[i,:], 44100)

def do_source_sep3():
    fnames = ["%s/mix%d.mat" % (data_dir,i) for i in range(1,4)]
    mixes = [io.loadmat(fname)[fname[4:-4]] for fname in fnames]
    do_source_separation(mixes)

def do_source_sep5():
    fname = "%s/mixFive.mat" % (data_dir)
    mixes = [io.loadmat(fname)["mixFive%d" % (i)] for i in range(1,6)]
    do_source_separation(mixes)

# do source separation on 3 sources
do_source_sep3()

# do source separation on 5 sources
do_source_sep5()