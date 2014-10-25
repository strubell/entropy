#
# hw4.py
#
# Running the experiments for hw4
#

from __future__ import division
import numpy as np
import image_alignment as align
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

data_dir = "dat"
fname = "00149v.jpg"
img = mpimg.imread(data_dir + "/" + fname)
plot = plt.imshow(img)
plt.show()
