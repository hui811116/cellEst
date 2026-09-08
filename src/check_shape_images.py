import numpy as np
import os
import sys
import glob
from PIL import Image

argv = sys.argv

img_path = argv[1]

shape_set = set()

for img_p in glob.glob(img_path+"/**/*.tif",recursive=True):
    # add shape
    img = Image.open(img_p)
    im_arr = np.array(img)
    shape_set.add(im_arr.shape)
print("Total number of shapes:{:}".format(len(shape_set)))
print(shape_set)