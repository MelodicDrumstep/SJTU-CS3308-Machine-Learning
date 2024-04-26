import os
import numpy as np

def get_data(
data_root: str,                             # file path of data to read
    )-> tuple:

    X_train = np.load(os.path.join(data_root,'X_train_sampled.npy'))    # load the training images (N,H,W)
    X_test = np.load(os.path.join(data_root,'X_test_sampled.npy'))      # load the testing images (N,H,W)
    Y_train = np.load(os.path.join(data_root,'y_train_sampled.npy'))    # load the training labels (N,)
    Y_test = np.load(os.path.join(data_root,'y_test_sampled.npy'))      # load the testing labels (N,)

    return X_train,X_test,Y_train,Y_test    # return the training and testing images and labels as a tuple

def standardize(
        H: np.ndarray,          # features to be standardized (n_samples, feature_dimensions)
        ) -> np.ndarray:

    mu = np.mean(H, axis=0)     # get mean value of each feature
    sigma = np.std(H, axis=0)  # get std value of each feature

    # Selects dimensions whose std is not 0 for normalization
    mask = sigma != 0
    H[:, mask] = (H[:, mask] - mu[mask]) / sigma[mask]

    return H

