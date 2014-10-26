#
# hw4.py
#
# Running the experiments for hw4
#

from __future__ import division
import image_alignment as imgalign
import matplotlib.image as mpimg
from scipy import misc as scimisc
#import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

data_dir = "dat"
processed_dir = "processed"
files = [f for f in listdir(data_dir) if isfile(join(data_dir,f))]
imgs = [mpimg.imread(data_dir + "/" + file) for file in files]
bins = [16, 32, 64, 128]

print imgs[1].shape
aligned_img = imgalign.align(imgs[1], 128)
scimisc.imsave(processed_dir + "/processed-" + files[1], aligned_img)
# plt.imshow(aligned_img)
# plt.show()

# for img in imgs:
#     for b in bins:
#         aligned_img = imgalign.align(img, b)
#            scimisc.imsave(processed_dir + "/processed-" + files[1], aligned_img)

