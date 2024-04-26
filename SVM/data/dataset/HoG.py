import os
import numpy as np

from skimage.feature import hog     #use "pip install scikit-image" to install the package
from tqdm import tqdm               #use "pip install tqdm" to install the package

def get_HOG(
    X: np.ndarray,                          # images to extract the HOG featues (N,H.W)
    )-> np.ndarray:

    print('*********** extract HoG features ***********')   # print info
    assert len(X.shape) == 3, 'the shape of the images should be (N,H,W)'
    H_list = []                             # create an empty list to store the features
    for index in tqdm(range(len(X))):       # loop to extract HOG features of each image
        hog_vector = hog(X[index], orientations=6, pixels_per_cell=(2, 2), cells_per_block=(1, 1), visualize=False) # extract the HOG features of each image(D,)
        H_list.append(hog_vector)           # append HOG features of each image
    H = np.array(H_list)                    # convert the list of the features to a numpy array

    return H                                # return the HOG features of the images (N,D)
