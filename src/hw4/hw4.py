#
# hw4.py
#
# Running the experiments for hw4
#

from __future__ import division
import numpy as np
import image_alignment as imgalign
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

data_dir = "dat"
fname = "00194v.jpg"
img = mpimg.imread(data_dir + "/" + fname)

b = 128
aligned_img = imgalign.align(img, b)
plot = plt.imshow(aligned_img)
plt.show()
