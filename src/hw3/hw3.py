#
# hw3.py
#
# Running the experiments for hw3
#
from __future__ import division
import numpy as np
import ica_tools as ica
import scipy.io as io
import scikits.audiolab as audio

data_dir = "dat/"
fnames = [data_dir + "mix1.mat", data_dir + "mix2.mat", data_dir + "mix3.mat"]

def get_matlab_array(fname):
    return io.loadmat(fname)[fname[4:-4]]
mixes = map(get_matlab_array, fnames)

# play audio
audio.play(mixes[0], fs=44100)
audio.play(mixes[1], fs=44100)
audio.play(mixes[2], fs=44100)

data = np.matrix(np.vstack(mixes))
#rotations = ica.transform(data)
rotations = np.concatenate(ica.transform(data), ica.transform(data))
final_rotation_matrix = reduce(np.dot, reversed(rotations))
print "final rotation matrix:"
print final_rotation_matrix

# play audio
audio.play(data[0,:], 44100)
audio.play(data[1,:], 44100)
audio.play(data[2,:], 44100)

