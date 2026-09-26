import numpy as np
from PIL import Image
import pathlib


def check_img_shapes(img_path):
    """Check the shapes of images in the given path and return a set of unique shapes."""
    shape_set = set()
    img_path = pathlib.Path(img_path)

    for img_p in img_path.rglob("*.tif"):
        # add shape
        img = Image.open(img_p)
        im_arr = np.array(img)
        shape_set.add(im_arr.shape)

    print("Total number of shapes: {:}".format(len(shape_set)))
    print(shape_set)
    return shape_set
